"""
Concrete Pipeline Stage Implementations
=======================================

Each stage encapsulates its configuration and execution logic,
eliminating the repeated pattern in the orchestrator.
"""

from pathlib import Path
from typing import List

from .base import PipelineStage, StageContext, StageResult


class SAMStage(PipelineStage):
    """Stage 1: SAM2 Segmentation."""

    @property
    def name(self) -> str:
        return "SAM2 Segmentation"

    @property
    def order(self) -> int:
        return 1

    @property
    def script_name(self) -> str:
        return "auto_roto.py"

    @property
    def is_critical(self) -> bool:
        return True  # Pipeline can't continue without initial segmentation

    def is_enabled(self, context: StageContext) -> bool:
        return not context.settings["skip_sam"]

    def build_args(self, context: StageContext, output_dir: Path) -> List[str]:
        args = [
            "--input", context.input_path,
            "--output", str(output_dir),
            "--sam-model", context.sam_model,
            "--format", context.output_format,
            "--bit-depth", str(context.bit_depth),
            "--no-refine",  # We do our own refinement
        ]

        # Add prompt type
        prompt = context.settings["prompt"]
        box = context.settings["box"]
        interactive = context.settings["interactive"]

        if prompt:
            args.extend(["--prompt", prompt])
        elif box:
            args.extend(["--box", box])
        elif interactive:
            args.append("--interactive")

        if context.verbose:
            args.append("--verbose")

        return args

    def get_fallback_output(self, context: StageContext) -> Path:
        # No fallback for SAM - it's the first stage
        return context.output_base / self.get_output_subdir()


class DepthStage(PipelineStage):
    """Stage 2: Depth Refinement."""

    @property
    def name(self) -> str:
        return "Depth Refinement"

    @property
    def order(self) -> int:
        return 2

    @property
    def script_name(self) -> str:
        return "depth_refine.py"

    def is_enabled(self, context: StageContext) -> bool:
        return not context.settings["skip_depth"]

    def build_args(self, context: StageContext, output_dir: Path) -> List[str]:
        # Find alpha source from SAM output
        alpha_source = self._find_alpha_source(context)

        args = [
            "--alpha", str(alpha_source),
            "--frames", context.input_path,
            "--output", str(output_dir),
            "--depth-model", context.depth_model,
            "--format", context.output_format,
            "--bit-depth", str(context.bit_depth),
        ]

        if context.verbose:
            args.extend(["--verbose", "--debug"])

        return args

    def _find_alpha_source(self, context: StageContext) -> Path:
        """Find the alpha directory from SAM output."""
        if context.sam_output:
            alpha_dir = context.sam_output / "alpha"
            if alpha_dir.exists():
                return alpha_dir
            return context.sam_output
        return context.output_base / "01_sam_segmentation" / "alpha"

    def get_fallback_output(self, context: StageContext) -> Path:
        return context.sam_output or context.output_base / "01_sam_segmentation"


class ViTMatteStage(PipelineStage):
    """Stage 3: ViTMatte Alpha Refinement."""

    @property
    def name(self) -> str:
        return "ViTMatte Alpha Refinement"

    @property
    def order(self) -> int:
        return 3

    @property
    def script_name(self) -> str:
        return "vitmatte_refine.py"

    def is_enabled(self, context: StageContext) -> bool:
        return not context.settings["skip_vitmatte"]

    def build_args(self, context: StageContext, output_dir: Path) -> List[str]:
        # Find sources
        sam_alpha = self._find_sam_alpha(context)
        depth_maps = self._find_depth_maps(context)

        args = [
            "--sam-mask", str(sam_alpha),
            "--depth", str(depth_maps),
            "--frames", context.input_path,
            "--output", str(output_dir),
            "--format", context.output_format,
            "--bit-depth", str(context.bit_depth),
            "--core-erosion", str(context.settings["core_erosion"]),
            "--adaptive-base", str(context.settings["vitmatte_adaptive_base"]),
            "--adaptive-max", str(context.settings["vitmatte_adaptive_max"]),
            "--save-trimap",
        ]

        if context.settings["vitmatte_motion_aware"]:
            args.append("--motion-aware")

        if context.verbose:
            args.append("--verbose")

        return args

    def _find_sam_alpha(self, context: StageContext) -> Path:
        if context.sam_output:
            alpha_dir = context.sam_output / "alpha"
            if alpha_dir.exists():
                return alpha_dir
            return context.sam_output
        return context.output_base / "01_sam_segmentation" / "alpha"

    def _find_depth_maps(self, context: StageContext) -> Path:
        if context.depth_output:
            depth_dir = context.depth_output / "depth"
            if depth_dir.exists():
                return depth_dir
            return context.depth_output
        return context.output_base / "02_depth_refinement" / "depth"

    def get_fallback_output(self, context: StageContext) -> Path:
        return context.depth_output or context.sam_output or context.output_base


class EdgeStage(PipelineStage):
    """Stage 4: Edge Refinement."""

    @property
    def name(self) -> str:
        return "Edge Refinement"

    @property
    def order(self) -> int:
        return 4

    @property
    def script_name(self) -> str:
        return "edge_refine.py"

    def is_enabled(self, context: StageContext) -> bool:
        return not context.settings["skip_edge"]

    def build_args(self, context: StageContext, output_dir: Path) -> List[str]:
        alpha_source = self._find_alpha_source(context)

        args = [
            "--alpha", str(alpha_source),
            "--output", str(output_dir),
            "--frames", context.input_path,
            "--softness", str(context.settings["edge_softness"]),
            "--core-erosion", str(context.settings["core_erosion"]),
            "--despill", str(context.settings["despill_strength"]),
            "--format", context.output_format,
            "--bit-depth", str(context.bit_depth),
        ]

        if context.verbose:
            args.append("--verbose")

        return args

    def _find_alpha_source(self, context: StageContext) -> Path:
        # Prefer ViTMatte output, then depth, then SAM
        for output in [context.vitmatte_output, context.depth_output, context.sam_output]:
            if output:
                alpha_dir = output / "alpha"
                if alpha_dir.exists():
                    return alpha_dir
                return output
        return context.get_alpha_source()

    def get_fallback_output(self, context: StageContext) -> Path:
        return context.vitmatte_output or context.depth_output or context.sam_output or context.output_base


class TemporalStage(PipelineStage):
    """Stage 5: Temporal Smoothing."""

    @property
    def name(self) -> str:
        return "Temporal Smoothing"

    @property
    def order(self) -> int:
        return 5

    @property
    def script_name(self) -> str:
        return "temporal_smooth.py"

    def is_enabled(self, context: StageContext) -> bool:
        return not context.settings["skip_temporal"]

    def build_args(self, context: StageContext, output_dir: Path) -> List[str]:
        alpha_source = self._find_alpha_source(context)

        args = [
            "--alpha", str(alpha_source),
            "--output", str(output_dir),
            "--frames", context.input_path,
            "--window", str(context.settings["temporal_window"]),
            "--keyframe-interval", str(context.settings["keyframe_interval"]),
            "--format", context.output_format,
            "--bit-depth", str(context.bit_depth),
        ]

        if context.verbose:
            args.append("--verbose")

        return args

    def _find_alpha_source(self, context: StageContext) -> Path:
        for output in [context.edge_output, context.vitmatte_output, context.depth_output]:
            if output:
                alpha_dir = output / "alpha"
                if alpha_dir.exists():
                    return alpha_dir
                return output
        return context.get_alpha_source()

    def get_fallback_output(self, context: StageContext) -> Path:
        return context.edge_output or context.vitmatte_output or context.depth_output or context.output_base


class CombineStage(PipelineStage):
    """Stage 6: Matte Combination."""

    @property
    def name(self) -> str:
        return "Matte Combination"

    @property
    def order(self) -> int:
        return 6

    @property
    def script_name(self) -> str:
        return "matte_combine.py"

    def is_enabled(self, context: StageContext) -> bool:
        return not context.settings["skip_combine"]

    def build_args(self, context: StageContext, output_dir: Path) -> List[str]:
        alpha_source = self._find_alpha_source(context)

        args = [
            "--alpha", str(alpha_source),
            "--output", str(output_dir),
            "--frames", context.input_path,
            "--core-erosion", str(context.settings["core_erosion"]),
            "--despill", str(context.settings["despill_strength"]),
            "--format", context.output_format,
            "--bit-depth", str(context.bit_depth),
        ]

        if context.verbose:
            args.append("--verbose")

        return args

    def _find_alpha_source(self, context: StageContext) -> Path:
        for output in [context.temporal_output, context.edge_output, context.vitmatte_output]:
            if output:
                alpha_dir = output / "alpha"
                if alpha_dir.exists():
                    return alpha_dir
                return output
        return context.get_alpha_source()

    def get_fallback_output(self, context: StageContext) -> Path:
        return context.temporal_output or context.edge_output or context.vitmatte_output or context.output_base


class HairStage(PipelineStage):
    """Stage 7: Hair Refinement (Optional)."""

    @property
    def name(self) -> str:
        return "Hair Refinement"

    @property
    def order(self) -> int:
        return 7

    @property
    def script_name(self) -> str:
        return "vitmatte_refine.py"

    def is_enabled(self, context: StageContext) -> bool:
        # Hair refinement is disabled by default
        return not context.settings["skip_hair"]

    def build_args(self, context: StageContext, output_dir: Path) -> List[str]:
        sam_alpha = self._find_sam_alpha(context)
        depth_maps = self._find_depth_maps(context)

        args = [
            "--sam-mask", str(sam_alpha),
            "--depth", str(depth_maps),
            "--frames", context.input_path,
            "--output", str(output_dir),
            "--format", context.output_format,
            "--bit-depth", str(context.bit_depth),
            "--adaptive-base", "2.0",
            "--adaptive-max", "60.0",
            "--motion-aware",
            "--save-trimap",
        ]

        return args

    def _find_sam_alpha(self, context: StageContext) -> Path:
        if context.sam_output:
            alpha_dir = context.sam_output / "alpha"
            if alpha_dir.exists():
                return alpha_dir
        return context.get_alpha_source()

    def _find_depth_maps(self, context: StageContext) -> Path:
        if context.depth_output:
            depth_dir = context.depth_output / "depth"
            if depth_dir.exists():
                return depth_dir
            return context.depth_output
        return context.output_base / "02_depth_refinement" / "depth"

    def get_fallback_output(self, context: StageContext) -> Path:
        return context.combine_output or context.temporal_output or context.output_base


def create_default_registry():
    """Create a registry with all default pipeline stages."""
    from .base import StageRegistry

    registry = StageRegistry()
    registry.register(SAMStage())
    registry.register(DepthStage())
    registry.register(ViTMatteStage())
    registry.register(EdgeStage())
    registry.register(TemporalStage())
    registry.register(CombineStage())
    registry.register(HairStage())

    return registry
