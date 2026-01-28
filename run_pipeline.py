#!/usr/bin/env python3
"""
AUTO-ROTO INTERACTIVE CLI
=========================

Comprehensive interactive launcher for the auto-roto pipeline.
Guides you through all available options with explanations.

Usage:
    python run_pipeline.py                    # Full interactive mode
    python run_pipeline.py --quick            # Quick mode (fewer questions)
    python run_pipeline.py --preset ultra     # Direct preset selection
    python run_pipeline.py --help-features    # Explain all features
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS & FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.RESET = ''

# Disable colors on Windows without proper terminal support
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        Colors.disable()

def c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"

def bold(text: str) -> str:
    return c(text, Colors.BOLD)

def dim(text: str) -> str:
    return c(text, Colors.DIM)

def success(text: str) -> str:
    return c(text, Colors.GREEN)

def warn(text: str) -> str:
    return c(text, Colors.YELLOW)

def error(text: str) -> str:
    return c(text, Colors.RED)

def info(text: str) -> str:
    return c(text, Colors.CYAN)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineOptions:
    """All configurable pipeline options."""
    # Input/Output
    input_path: str = ""
    output_path: str = "./output"
    clean_output: bool = False
    
    # Detection mode
    prompt: str = "person"
    prompt_mode: str = "text"  # text, box, point, interactive, interactive_points
    box_coords: str = ""       # "x1,y1,x2,y2"
    point_coords: str = ""     # "x,y"
    interactive_points: bool = False  # Include/exclude point marking mode
    
    # Quality
    quality: str = "standard"
    refiner: str = "vitmatte"  # vitmatte or ma1
    
    # SAM settings
    sam_conf: float = 0.15
    sam_imgsz: int = 0  # 0 = auto
    
    # Depth settings
    depth_model: str = "large"
    depth_res: Optional[int] = None
    depth_method: str = "upper"
    depth_percentiles: Tuple[float, float] = (2.0, 98.0)
    depth_first: bool = False
    use_depth_confidence: bool = False
    
    # Point cloud
    export_pointcloud: bool = False
    pointcloud_format: str = "ply"
    
    # Hair refinement
    with_hair: bool = False
    
    # Stages to skip
    skip_sam: bool = False
    skip_depth: bool = False
    skip_vitmatte: bool = False
    # Edge refinement and temporal smoothing removed from pipeline
    skip_combine: bool = False


QUALITY_PRESETS = {
    "draft": {
        "description": "Fast preview - lower quality, quick iteration",
        "time": "~2-5 min",
        "quality": "draft",
        "refiner": "vitmatte",
        "sam_conf": 0.25,
        "depth_res": None,
        "depth_method": "upper",
        "depth_percentiles": (2.0, 98.0),
    },
    "standard": {
        "description": "Balanced quality and speed - good for most cases",
        "time": "~5-10 min",
        "quality": "standard",
        "refiner": "vitmatte",
        "sam_conf": 0.15,
        "depth_res": 1536,
        "depth_method": "upper",
        "depth_percentiles": (2.0, 98.0),
    },
    "high": {
        "description": "High quality - fine detail preservation",
        "time": "~10-20 min",
        "quality": "high",
        "refiner": "ma1",
        "sam_conf": 0.15,
        "depth_res": 2048,
        "depth_method": "lower",
        "depth_percentiles": (1.0, 99.0),
    },
    "ultra": {
        "description": "Ultra quality with MatAnyone refiner",
        "time": "~20-40 min",
        "quality": "ultra",
        "refiner": "ma1",
        "sam_conf": 0.10,
        "depth_res": 2048,
        "depth_method": "lower",
        "depth_percentiles": (0.0, 100.0),
    },
    "ultra_detail": {
        "description": "Ultra + aggressive SAM for thin objects (drawstrings)",
        "time": "~25-45 min",
        "quality": "ultra",
        "refiner": "ma1",
        "sam_conf": 0.08,
        "depth_res": 2048,
        "depth_method": "lower",
        "depth_percentiles": (0.0, 100.0),
        "use_depth_confidence": True,
    },
    "ultra_hair": {
        "description": "Ultra + hair refinement stage for wispy edges",
        "time": "~30-50 min",
        "quality": "ultra",
        "refiner": "ma1",
        "sam_conf": 0.08,
        "depth_res": 2048,
        "depth_method": "lower",
        "depth_percentiles": (0.0, 100.0),
        "use_depth_confidence": True,
        "with_hair": True,
    },
    "ultra_depth": {
        "description": "Ultra + pure depth FIRST (no alpha input to depth)",
        "time": "~25-45 min",
        "quality": "ultra",
        "refiner": "ma1",
        "sam_conf": 0.10,
        "depth_res": 2048,
        "depth_method": "lower",
        "depth_percentiles": (0.0, 100.0),
        "depth_first": True,
    },
    "ultra_3d": {
        "description": "Ultra + depth first + point cloud export (PLY)",
        "time": "~30-50 min",
        "quality": "ultra",
        "refiner": "ma1",
        "sam_conf": 0.10,
        "depth_res": 2048,
        "depth_method": "lower",
        "depth_percentiles": (0.0, 100.0),
        "depth_first": True,
        "export_pointcloud": True,
        "pointcloud_format": "ply",
    },
    "max_interactive": {
        "description": "MAX QUALITY + interactive include/exclude marking",
        "time": "~30-60 min",
        "quality": "ultra",
        "refiner": "ma1",
        "sam_conf": 0.08,
        "depth_res": 2048,
        "depth_method": "lower",
        "depth_percentiles": (0.0, 100.0),
        "use_depth_confidence": True,
        "interactive_points": True,  # Special flag for point-based include/exclude
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print()
    print(bold("╔══════════════════════════════════════════════════════════════╗"))
    print(bold("║") + info("            AUTO-ROTO INTERACTIVE PIPELINE              ") + bold("║"))
    print(bold("║") + dim("         Professional Video Rotoscoping Tool             ") + bold("║"))
    print(bold("╚══════════════════════════════════════════════════════════════╝"))
    print()

def print_section(title: str):
    print()
    print(bold(f"┌─ {title} " + "─" * (55 - len(title)) + "┐"))

def print_option(num: int, name: str, desc: str, selected: bool = False):
    marker = success("●") if selected else dim("○")
    print(f"  {marker} [{num}] {bold(name):20} {dim(desc)}")

def print_info_box(lines: List[str]):
    print()
    print(dim("  ┌" + "─" * 56 + "┐"))
    for line in lines:
        print(dim(f"  │ {line:<54} │"))
    print(dim("  └" + "─" * 56 + "┘"))

def prompt_choice(prompt: str, valid: List[str], default: str = None) -> str:
    """Get validated user input."""
    default_hint = f" [{default}]" if default else ""
    while True:
        try:
            choice = input(f"  {prompt}{default_hint}: ").strip()
            if not choice and default:
                return default
            if choice.lower() in [v.lower() for v in valid]:
                return choice.lower()
            # Try numeric
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(valid):
                    return valid[idx - 1]
            print(error(f"  Invalid choice. Options: {', '.join(valid)}"))
        except KeyboardInterrupt:
            print("\n\n  Cancelled by user.")
            sys.exit(0)

def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        choice = input(f"  {prompt} {hint}: ").strip().lower()
        if not choice:
            return default
        return choice in ('y', 'yes', 'true', '1')
    except KeyboardInterrupt:
        print("\n\n  Cancelled by user.")
        sys.exit(0)

def prompt_string(prompt: str, default: str = "") -> str:
    """Get string input with optional default."""
    hint = f" [{default}]" if default else ""
    try:
        value = input(f"  {prompt}{hint}: ").strip()
        return value if value else default
    except KeyboardInterrupt:
        print("\n\n  Cancelled by user.")
        sys.exit(0)

def prompt_path(prompt: str, must_exist: bool = True, default: str = "") -> str:
    """Get and validate a path."""
    while True:
        path = prompt_string(prompt, default)
        if not path:
            print(error("  Path cannot be empty."))
            continue
        if must_exist and not Path(path).exists():
            print(error(f"  Path not found: {path}"))
            continue
        return path

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def select_input(opts: PipelineOptions) -> PipelineOptions:
    """Select input source."""
    print_section("INPUT SOURCE")
    print()
    
    # Find existing directories
    script_dir = Path(__file__).parent
    candidates = []
    for name in ["test_input", "input", "frames", "source"]:
        p = script_dir / name
        if p.exists() and p.is_dir():
            count = len(list(p.glob("*")))
            candidates.append((name, f"{count} files"))
    
    if candidates:
        print("  Found directories:")
        for i, (name, info) in enumerate(candidates, 1):
            print(f"    [{i}] ./{name}/ {dim(f'({info})')}")
        print(f"    [{len(candidates)+1}] Custom path...")
        print()
        
        choice = prompt_string("Select input", "1")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                opts.input_path = f"./{candidates[idx][0]}"
                print(success(f"  ✓ Using: {opts.input_path}"))
                return opts
        
    # Custom path
    opts.input_path = prompt_path("Enter input path (frames dir or video)")
    print(success(f"  ✓ Using: {opts.input_path}"))
    return opts


def select_detection_mode(opts: PipelineOptions) -> PipelineOptions:
    """Select how to detect the subject."""
    print_section("DETECTION MODE")
    print()
    
    print_info_box([
        "How should the pipeline find your subject?",
        "",
        "• TEXT: Describe what to find (e.g., 'person', 'dog')",
        "• INTERACTIVE: Draw a box on the first frame",
        "• BOX: Specify exact coordinates",
        "• POINT: Click on a specific location",
    ])
    print()
    
    print_option(1, "Text Prompt", "Describe what to segment (recommended)")
    print_option(2, "Interactive", "Draw a box visually on first frame")
    print_option(3, "Box Coords", "Specify x1,y1,x2,y2 coordinates")
    print_option(4, "Point", "Click on a specific point")
    print()
    
    choice = prompt_choice("Select mode", ["1", "2", "3", "4"], "1")
    
    if choice == "1":
        opts.prompt_mode = "text"
        print()
        print_info_box([
            "Enter what to detect. Use '.' to separate multiple objects.",
            "Examples: 'person' | 'person.dog' | 'man in red shirt'",
        ])
        opts.prompt = prompt_string("What to detect?", "person")
        
    elif choice == "2":
        opts.prompt_mode = "interactive"
        print(info("\n  A window will open to draw boxes on the first frame."))
        
    elif choice == "3":
        opts.prompt_mode = "box"
        print()
        print_info_box([
            "Enter bounding box as: x1,y1,x2,y2",
            "Example: 100,50,400,500",
        ])
        opts.box_coords = prompt_string("Box coordinates")
        
    elif choice == "4":
        opts.prompt_mode = "point"
        print()
        print_info_box([
            "Enter point as: x,y",
            "Example: 250,300",
        ])
        opts.point_coords = prompt_string("Point coordinates")
    
    print(success(f"  ✓ Mode: {opts.prompt_mode}"))
    return opts


def select_quality(opts: PipelineOptions) -> PipelineOptions:
    """Select quality preset."""
    print_section("QUALITY PRESET")
    print()
    
    print_info_box([
        "Quality presets balance speed vs. detail preservation.",
        "Higher quality = longer processing time.",
        "Specialized presets available for hair, depth, 3D export.",
    ])
    print()
    
    # Group presets for display
    basic_presets = ["draft", "standard", "high", "ultra"]
    special_presets = ["ultra_detail", "ultra_hair", "ultra_depth", "ultra_3d", "max_interactive"]
    
    print(dim("  Basic Presets:"))
    for i, name in enumerate(basic_presets, 1):
        cfg = QUALITY_PRESETS[name]
        time_info = dim(f"({cfg['time']})")
        print_option(i, name.upper(), f"{cfg['description']} {time_info}")
    
    print()
    print(dim("  Specialized Presets:"))
    for i, name in enumerate(special_presets, len(basic_presets) + 1):
        cfg = QUALITY_PRESETS[name]
        time_info = dim(f"({cfg['time']})")
        print_option(i, name.upper(), f"{cfg['description']} {time_info}")
    print()
    
    all_presets = basic_presets + special_presets
    choice = prompt_choice("Select preset (1-9 or name)", all_presets, "standard")
    
    # Apply preset values
    preset = QUALITY_PRESETS[choice]
    opts.quality = preset["quality"]
    opts.refiner = preset["refiner"]
    opts.sam_conf = preset["sam_conf"]
    opts.depth_res = preset["depth_res"]
    opts.depth_method = preset["depth_method"]
    opts.depth_percentiles = preset["depth_percentiles"]
    
    # Apply optional preset features
    if preset.get("depth_first"):
        opts.depth_first = True
    if preset.get("use_depth_confidence"):
        opts.use_depth_confidence = True
    if preset.get("with_hair"):
        opts.with_hair = True
    if preset.get("export_pointcloud"):
        opts.export_pointcloud = True
        opts.pointcloud_format = preset.get("pointcloud_format", "ply")
    if preset.get("interactive_points"):
        opts.interactive_points = True
        opts.prompt_mode = "interactive_points"
    
    print(success(f"  ✓ Preset: {choice.upper()}"))
    
    # If interactive points mode, explain it
    if opts.interactive_points:
        print()
        print_info_box([
            "INTERACTIVE POINTS MODE:",
            "  • LEFT CLICK = INCLUDE (foreground)",
            "  • RIGHT CLICK = EXCLUDE (background)",
            "  • Press ENTER when done marking",
            "",
            "Mark what you WANT, then what you DON'T want.",
        ])
    
    return opts


def select_advanced_options(opts: PipelineOptions) -> PipelineOptions:
    """Configure advanced options."""
    print_section("ADVANCED OPTIONS")
    print()
    
    if not prompt_yes_no("Configure advanced options?", default=False):
        print(dim("  Skipping advanced options."))
        return opts
    
    # ─── SAM Settings ───
    print()
    print(bold("  SAM Segmentation:"))
    print_info_box([
        "SAM Confidence: Lower = more detections, may include noise",
        "  • 0.25 = Fast, may miss fine details",
        "  • 0.15 = Balanced (default)",
        "  • 0.08 = Aggressive, catches thin objects like drawstrings",
    ])
    
    conf_str = prompt_string(f"SAM confidence (0.01-1.0)", str(opts.sam_conf))
    try:
        opts.sam_conf = float(conf_str)
    except ValueError:
        pass
    
    # ─── Depth Settings ───
    print()
    print(bold("  Depth Estimation:"))
    
    if prompt_yes_no("Run pure depth FIRST (before SAM)?", default=False):
        opts.depth_first = True
        print(info("  Pure depth will run on RGB only, no alpha influence."))
    
    if prompt_yes_no("Export point cloud from depth?", default=False):
        opts.export_pointcloud = True
        opts.pointcloud_format = prompt_choice(
            "Point cloud format", ["ply", "glb"], "ply"
        )
    
    if prompt_yes_no("Use depth confidence for semi-transparent edges?", default=False):
        opts.use_depth_confidence = True
    
    # ─── Hair Refinement ───
    print()
    print(bold("  Hair Refinement:"))
    print_info_box([
        "Extra stage for fine hair/fur detail.",
        "Increases processing time but improves wispy edges.",
    ])
    
    if prompt_yes_no("Enable hair refinement stage?", default=False):
        opts.with_hair = True
    
    # ─── Clean Output ───
    print()
    print(bold("  Output Handling:"))
    if prompt_yes_no("Clear output directories before running (--clean)?", default=True):
        opts.clean_output = True
    
    return opts


def select_output(opts: PipelineOptions) -> PipelineOptions:
    """Select output directory."""
    print_section("OUTPUT")
    print()
    
    opts.output_path = prompt_string("Output directory", "./output")
    
    # Check if exists
    out_path = Path(opts.output_path)
    if out_path.exists():
        final_dir = out_path / "final"
        if final_dir.exists():
            count = len(list(final_dir.glob("*/*")))
            if count > 0:
                print(warn(f"  ⚠ Output exists with {count} files"))
                if not opts.clean_output:
                    opts.clean_output = prompt_yes_no("Clean existing output?", default=True)
    
    print(success(f"  ✓ Output: {opts.output_path}"))
    return opts


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_command(opts: PipelineOptions) -> List[str]:
    """Build the full pipeline command from options."""
    cmd = [
        sys.executable, "full_pipeline_v5.py",
        "--input", opts.input_path,
        "--output", opts.output_path,
        "--quality", opts.quality,
        "--refiner", opts.refiner,
        "--sam-conf", str(opts.sam_conf),
        "--depth-method", opts.depth_method,
        "--depth-percentiles", 
        str(opts.depth_percentiles[0]), 
        str(opts.depth_percentiles[1]),
    ]
    
    # Detection mode
    if opts.prompt_mode == "text":
        cmd.extend(["--prompt", opts.prompt])
    elif opts.prompt_mode == "interactive":
        cmd.append("--interactive")
    elif opts.prompt_mode == "interactive_points":
        cmd.append("--interactive-points")
    elif opts.prompt_mode == "box":
        cmd.extend(["--box", opts.box_coords])
    elif opts.prompt_mode == "point":
        cmd.extend(["--point", opts.point_coords])
    
    # Depth settings
    if opts.depth_res:
        cmd.extend(["--depth-res", str(opts.depth_res)])
    if opts.depth_first:
        cmd.append("--depth-first")
    if opts.use_depth_confidence:
        cmd.append("--use-depth-confidence")
    
    # Point cloud
    if opts.export_pointcloud:
        cmd.append("--export-pointcloud")
        cmd.extend(["--pointcloud-format", opts.pointcloud_format])
    
    # Hair
    if opts.with_hair:
        cmd.append("--with-hair")
    
    # Clean
    if opts.clean_output:
        cmd.append("--clean")
    
    return cmd


def print_summary(opts: PipelineOptions, cmd: List[str]):
    """Print configuration summary."""
    print()
    print(bold("╔══════════════════════════════════════════════════════════════╗"))
    print(bold("║") + info("                    CONFIGURATION SUMMARY                  ") + bold("║"))
    print(bold("╚══════════════════════════════════════════════════════════════╝"))
    print()
    
    # Input/Output
    print(f"  {bold('Input:')}        {opts.input_path}")
    print(f"  {bold('Output:')}       {opts.output_path}")
    print(f"  {bold('Clean:')}        {'Yes' if opts.clean_output else 'No'}")
    print()
    
    # Detection
    print(f"  {bold('Detection:')}    {opts.prompt_mode.upper()}")
    if opts.prompt_mode == "text":
        print(f"  {bold('Prompt:')}       \"{opts.prompt}\"")
    elif opts.prompt_mode == "box":
        print(f"  {bold('Box:')}          {opts.box_coords}")
    elif opts.prompt_mode == "point":
        print(f"  {bold('Point:')}        {opts.point_coords}")
    print()
    
    # Quality
    print(f"  {bold('Quality:')}      {opts.quality.upper()}")
    print(f"  {bold('Refiner:')}      {opts.refiner}")
    print(f"  {bold('SAM Conf:')}     {opts.sam_conf}")
    print()
    
    # Depth
    print(f"  {bold('Depth Res:')}    {opts.depth_res or 'auto'}")
    print(f"  {bold('Depth First:')}  {'Yes' if opts.depth_first else 'No'}")
    if opts.export_pointcloud:
        print(f"  {bold('Point Cloud:')} {opts.pointcloud_format.upper()}")
    print()
    
    # Extras
    extras = []
    if opts.with_hair:
        extras.append("Hair Refinement")
    if opts.use_depth_confidence:
        extras.append("Depth Confidence")
    if extras:
        print(f"  {bold('Extras:')}       {', '.join(extras)}")
        print()
    
    # Command
    print(dim("  Command:"))
    print(dim("  " + " ".join(cmd[:6])))
    print(dim("    " + " ".join(cmd[6:])))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE FLOW
# ═══════════════════════════════════════════════════════════════════════════════

def run_interactive(quick_mode: bool = False):
    """Run the full interactive experience."""
    print_header()
    
    opts = PipelineOptions()
    
    # Step through configuration
    opts = select_input(opts)
    opts = select_detection_mode(opts)
    opts = select_quality(opts)
    
    if not quick_mode:
        opts = select_advanced_options(opts)
    
    opts = select_output(opts)
    
    # Build and show command
    cmd = build_command(opts)
    print_summary(opts, cmd)
    
    # Confirm and run
    if prompt_yes_no("Start pipeline?", default=True):
        print()
        print(bold("═" * 62))
        print(success("  STARTING PIPELINE..."))
        print(bold("═" * 62))
        print()
        
        script_dir = Path(__file__).parent
        result = subprocess.run(cmd, cwd=script_dir)
        
        print()
        if result.returncode == 0:
            print(success("  ✓ Pipeline completed successfully!"))
            print(f"    Output: {opts.output_path}/final/")
        else:
            print(error(f"  ✗ Pipeline failed with code {result.returncode}"))
        
        return result.returncode
    else:
        print(dim("  Cancelled."))
        return 0


def print_features_help():
    """Print detailed explanation of all features."""
    print_header()
    print(bold("\n  FEATURE REFERENCE"))
    print("  " + "=" * 56)
    
    print(bold("\n  DETECTION MODES"))
    print("  " + "-" * 56)
    print("""
  TEXT PROMPT (--prompt "person")
    Describe what to find. SAM3 uses built-in text understanding.
    Examples: "person", "dog", "man in red shirt", "person.car"
    
  INTERACTIVE (--interactive)
    Opens a window to draw bounding boxes on the first frame.
    Great for precise selection of specific subjects.
    
  BOX COORDS (--box "x1,y1,x2,y2")
    Specify exact pixel coordinates for detection area.
    
  POINT (--point "x,y")
    Click on a specific location to segment that object.
    """)
    
    print(bold("\n  QUALITY SETTINGS"))
    print("  " + "-" * 56)
    print("""
  DRAFT
    Fast preview. Good for checking if detection works.
    
  STANDARD  
    Balanced quality/speed. Good for most cases.
    
  HIGH
    Fine detail preservation. Uses MatAnyone1 refiner.
    
  ULTRA
    Maximum quality. All settings optimized for best output.
    """)
    
    print(bold("\n  DEPTH OPTIONS"))
    print("  " + "-" * 56)
    print("""
  --depth-first
    Run pure depth estimation BEFORE SAM segmentation.
    Depth is computed on RGB only, no alpha influence.
    Better for getting clean, unbiased depth maps.
    
  --export-pointcloud
    Export 3D point cloud (PLY or GLB format) from depth.
    Useful for 3D visualization or further processing.
    
  --use-depth-confidence
    Use DA3's confidence maps for semi-transparent edges.
    Can improve handling of glass, smoke, thin objects.
    """)
    
    print(bold("\n  OTHER OPTIONS"))
    print("  " + "-" * 56)
    print("""
  --with-hair
    Enable extra hair refinement stage.
    Better results for fine hair, fur, and wispy edges.
    
  --clean
    Clear all output directories before running.
    Removes stale files from previous runs.
    
  --sam-conf <0.01-1.0>
    SAM confidence threshold. Lower = more detections.
    Try 0.08 for thin objects like drawstrings.
    """)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Auto-Roto Interactive Pipeline Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    python run_pipeline.py                     # Full interactive
    python run_pipeline.py --quick             # Quick mode
    python run_pipeline.py --help-features     # Feature reference
    python run_pipeline.py -i ./frames -p ultra --dry-run
        """
    )
    
    # Modes
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick mode (skip advanced options)")
    parser.add_argument("--help-features", "-H", action="store_true",
                       help="Show detailed feature explanations")
    
    # Direct options
    parser.add_argument("--preset", "-p", choices=list(QUALITY_PRESETS.keys()),
                       help="Quality preset")
    parser.add_argument("--input", "-i", help="Input path")
    parser.add_argument("--prompt", default="person", help="Detection prompt")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show command without running")
    
    args = parser.parse_args()
    
    # Feature help
    if args.help_features:
        print_features_help()
        return 0
    
    # Direct mode with all args
    if args.preset and args.input:
        opts = PipelineOptions(
            input_path=args.input,
            output_path=args.output,
            prompt=args.prompt,
            clean_output=True,
        )
        
        # Apply preset
        preset = QUALITY_PRESETS[args.preset]
        opts.quality = preset["quality"]
        opts.refiner = preset["refiner"]
        opts.sam_conf = preset["sam_conf"]
        opts.depth_res = preset["depth_res"]
        opts.depth_method = preset["depth_method"]
        opts.depth_percentiles = preset["depth_percentiles"]
        
        cmd = build_command(opts)
        
        if args.dry_run:
            print("Command:")
            print(" ".join(cmd))
            return 0
        
        script_dir = Path(__file__).parent
        result = subprocess.run(cmd, cwd=script_dir)
        return result.returncode
    
    # Interactive mode
    return run_interactive(quick_mode=args.quick)


if __name__ == "__main__":
    sys.exit(main())
