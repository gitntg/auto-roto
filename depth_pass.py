#!/usr/bin/env python3
"""
PURE DEPTH ESTIMATION PASS
==========================

Standalone depth estimation using Depth Anything 3 (DA3).
NO alpha/mask input - pure RGB to depth conversion.

Outputs:
    - Raw depth maps (EXR float32)
    - Depth visualizations (PNG with colormap)
    - Optional: Point cloud (PLY/GLB)

USAGE:
    # Basic depth estimation
    python depth_pass.py --frames ./frames/ --output ./depth_output/

    # With point cloud export
    python depth_pass.py --frames ./frames/ --output ./depth_output/ --export-pointcloud

    # High resolution processing
    python depth_pass.py --frames ./frames/ --output ./depth_output/ --process-res 2048

    # From video
    python depth_pass.py --video input.mp4 --output ./depth_output/
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import cv2
import torch


@dataclass
class DepthPassConfig:
    """Configuration for pure depth estimation."""
    
    # Input
    frames_dir: str = ""
    video_path: str = ""
    
    # Output
    output_dir: str = "./depth_output"
    
    # Model settings
    model_size: str = "large"  # small, base, large, nested-large
    
    # Processing settings
    process_res: Optional[int] = None  # None = auto (use image size)
    process_method: str = "upper"      # "upper" or "lower" bound resize
    
    # Normalization
    norm_percentiles: Tuple[float, float] = (2.0, 98.0)
    
    # Export options
    export_pointcloud: bool = False
    pointcloud_format: str = "ply"  # "ply" or "glb"
    max_pointcloud_points: int = 1_000_000
    
    # Output format
    save_exr: bool = True
    save_visualization: bool = True
    
    # Performance
    device: str = "cuda"
    
    # Debug
    verbose: bool = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("DepthPass")


MODEL_NAME_BY_SIZE = {
    "small": "depth-anything/DA3-Small",
    "base": "depth-anything/DA3-Base",
    "large": "depth-anything/DA3Mono-Large",
    "nested-large": "depth-anything/DA3NESTED-GIANT-LARGE",
    "nested-base": "depth-anything/DA3NESTED-Base",
}

MAX_PROCESS_RES = 8192


class DepthEstimator:
    """Pure depth estimation using Depth Anything 3."""
    
    def __init__(
        self,
        model_size: str = 'large',
        device: str = 'cuda',
        process_res: Optional[int] = None,
        process_method: str = "upper",
        norm_percentiles: Tuple[float, float] = (2.0, 98.0),
        logger: logging.Logger = None
    ):
        self.model_size = model_size
        self.device = device
        self.process_res = process_res
        self.process_method = process_method
        self.norm_percentiles = norm_percentiles
        self.logger = logger or logging.getLogger("DepthEstimator")
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Depth Anything 3 model."""
        model_name = MODEL_NAME_BY_SIZE.get(self.model_size, MODEL_NAME_BY_SIZE["large"])
        self.logger.info(f"Loading Depth Anything 3 ({model_name})...")
        
        # Add DA3 repo to path
        script_dir = Path(__file__).parent
        depth_repo = script_dir / "depth-anything-3"
        if depth_repo.exists() and str(depth_repo) not in sys.path:
            sys.path.insert(0, str(depth_repo))
        
        from depth_anything_3.api import DepthAnything3
        
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()
        
        self.logger.info("Depth Anything 3 loaded successfully")
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Args:
            image: RGB image (H, W, 3), uint8 or float
            
        Returns:
            Depth map (H, W), float32, normalized 0-1
            Closer objects = LOWER values (DA3 convention)
        """
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        h, w = image.shape[:2]
        
        # Determine processing resolution
        if self.process_res is not None:
            proc_res = min(self.process_res, MAX_PROCESS_RES)
        else:
            proc_res = min(max(h, w), MAX_PROCESS_RES)
        
        proc_method = f"{self.process_method}_bound_resize"
        
        self.logger.debug(f"Estimating depth: {w}x{h} -> process_res={proc_res}, method={proc_method}")
        
        with torch.no_grad():
            prediction = self.model.inference(
                [image],
                process_res=proc_res,
                process_res_method=proc_method,
            )
        
        depth = prediction.depth[0]
        
        # Resize to match input
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize using percentiles
        p_low, p_high = np.percentile(depth, list(self.norm_percentiles))
        depth = np.clip((depth - p_low) / (p_high - p_low + 1e-8), 0, 1)
        
        return depth.astype(np.float32)
    
    def estimate_with_prediction(self, image: np.ndarray):
        """
        Estimate depth and return full prediction object (for point cloud export).
        
        Returns:
            (depth_normalized, prediction_object)
        """
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        h, w = image.shape[:2]
        
        if self.process_res is not None:
            proc_res = min(self.process_res, MAX_PROCESS_RES)
        else:
            proc_res = min(max(h, w), MAX_PROCESS_RES)
        
        proc_method = f"{self.process_method}_bound_resize"
        
        with torch.no_grad():
            prediction = self.model.inference(
                [image],
                process_res=proc_res,
                process_res_method=proc_method,
            )
        
        depth = prediction.depth[0]
        
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        p_low, p_high = np.percentile(depth, list(self.norm_percentiles))
        depth_norm = np.clip((depth - p_low) / (p_high - p_low + 1e-8), 0, 1)
        
        return depth_norm.astype(np.float32), prediction
    
    def release(self):
        """Release model and free GPU memory."""
        if self.model is not None:
            try:
                self.model = self.model.to('cpu')
            except Exception:
                pass
            del self.model
            self.model = None
        
        try:
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


def save_depth_exr(depth: np.ndarray, filepath: Path):
    """Save depth as float32 EXR."""
    try:
        import OpenEXR
        import Imath
        
        h, w = depth.shape
        header = OpenEXR.Header(w, h)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        header['channels'] = {'Y': Imath.Channel(pixel_type)}
        
        exr = OpenEXR.OutputFile(str(filepath), header)
        exr.writePixels({'Y': depth.astype(np.float32).tobytes()})
        exr.close()
        
    except ImportError:
        # Fallback to 16-bit PNG
        depth_16 = (depth * 65535).astype(np.uint16)
        cv2.imwrite(str(filepath).replace('.exr', '.png'), depth_16)


def save_depth_visualization(depth: np.ndarray, filepath: Path):
    """Save depth as colorized PNG visualization."""
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('inferno')
        depth_colored = cmap(depth)
        depth_vis = (depth_colored[:, :, :3] * 255).astype(np.uint8)
        depth_vis = depth_vis[:, :, ::-1]  # RGB to BGR
    except ImportError:
        depth_vis = (depth * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    
    cv2.imwrite(str(filepath), depth_vis)


def load_frames(frames_dir: Path) -> List[Tuple[str, np.ndarray]]:
    """Load all frames from directory."""
    extensions = {'.png', '.jpg', '.jpeg', '.exr', '.tif', '.tiff'}
    frame_files = sorted([
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in extensions
    ])
    
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append((f.stem, img))
    
    return frames


def extract_frames_from_video(video_path: Path, output_dir: Path) -> List[Tuple[str, np.ndarray]]:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((f"frame_{idx:06d}", frame_rgb))
        idx += 1
    
    cap.release()
    return frames


class DepthPassPipeline:
    """Pipeline for pure depth estimation."""
    
    def __init__(self, config: DepthPassConfig):
        self.config = config
        self.logger = setup_logging(config.verbose)
        self.estimator = None
    
    def run(self):
        """Run the depth pass pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("PURE DEPTH ESTIMATION PASS")
        self.logger.info("=" * 60)
        
        # Setup output directories
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        depth_dir = output_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        
        vis_dir = output_dir / "depth_vis"
        if self.config.save_visualization:
            vis_dir.mkdir(exist_ok=True)
        
        # Load frames
        if self.config.video_path:
            self.logger.info(f"Extracting frames from video: {self.config.video_path}")
            frames = extract_frames_from_video(Path(self.config.video_path), output_dir)
        elif self.config.frames_dir:
            self.logger.info(f"Loading frames from: {self.config.frames_dir}")
            frames = load_frames(Path(self.config.frames_dir))
        else:
            self.logger.error("No input specified (--frames or --video)")
            return False
        
        self.logger.info(f"Loaded {len(frames)} frames")
        
        if not frames:
            self.logger.error("No frames found")
            return False
        
        # Initialize depth estimator
        self.logger.info(f"Model: {self.config.model_size}")
        self.logger.info(f"Process resolution: {self.config.process_res or 'auto'}")
        self.logger.info(f"Process method: {self.config.process_method}")
        
        self.estimator = DepthEstimator(
            model_size=self.config.model_size,
            device=self.config.device,
            process_res=self.config.process_res,
            process_method=self.config.process_method,
            norm_percentiles=self.config.norm_percentiles,
            logger=self.logger
        )
        
        # Process frames
        predictions = []
        for idx, (name, frame) in enumerate(frames):
            self.logger.info(f"Processing {idx+1}/{len(frames)}: {name}")
            
            if self.config.export_pointcloud:
                depth, prediction = self.estimator.estimate_with_prediction(frame)
                predictions.append(prediction)
            else:
                depth = self.estimator.estimate(frame)
            
            # Save depth EXR
            if self.config.save_exr:
                exr_path = depth_dir / f"{name}.exr"
                save_depth_exr(depth, exr_path)
            
            # Save visualization
            if self.config.save_visualization:
                vis_path = vis_dir / f"{name}.png"
                save_depth_visualization(depth, vis_path)
        
        # Export point cloud if requested
        if self.config.export_pointcloud and predictions:
            self._export_pointcloud(predictions, output_dir)
        
        # Cleanup
        self.estimator.release()
        
        self.logger.info("=" * 60)
        self.logger.info("DEPTH PASS COMPLETE")
        self.logger.info(f"Output: {output_dir}")
        self.logger.info("=" * 60)
        
        return True
    
    def _export_pointcloud(self, predictions, output_dir: Path):
        """Export point cloud from predictions."""
        self.logger.info("Exporting point cloud...")
        
        try:
            from depth_anything_3.utils.export import export
            
            pc_dir = output_dir / "pointcloud"
            pc_dir.mkdir(exist_ok=True)
            
            # Export each prediction
            for idx, pred in enumerate(predictions):
                if self.config.pointcloud_format == "ply":
                    # Check if we have Gaussian splatting data
                    if hasattr(pred, 'gaussians') and pred.gaussians is not None:
                        export(pred, "gs_ply", str(pc_dir))
                        self.logger.info(f"  Exported Gaussian PLY for frame {idx}")
                    else:
                        self.logger.warning(f"  Frame {idx}: No Gaussian data for PLY export")
                
                elif self.config.pointcloud_format == "glb":
                    if (hasattr(pred, 'processed_images') and pred.processed_images is not None and
                        hasattr(pred, 'intrinsics') and pred.intrinsics is not None and
                        hasattr(pred, 'extrinsics') and pred.extrinsics is not None):
                        export(pred, "glb", str(pc_dir), glb={
                            "num_max_points": self.config.max_pointcloud_points
                        })
                        self.logger.info(f"  Exported GLB for frame {idx}")
                    else:
                        self.logger.warning(f"  Frame {idx}: Missing data for GLB export (need intrinsics/extrinsics)")
            
            self.logger.info(f"Point cloud export complete: {pc_dir}")
            
        except ImportError as e:
            self.logger.error(f"Point cloud export failed: {e}")
        except Exception as e:
            self.logger.error(f"Point cloud export error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure Depth Estimation Pass using Depth Anything 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Basic depth from frames
    %(prog)s --frames ./frames/ --output ./depth_output/
    
    # From video with high resolution
    %(prog)s --video input.mp4 --output ./depth_output/ --process-res 2048
    
    # With point cloud export
    %(prog)s --frames ./frames/ --output ./depth_output/ --export-pointcloud --pointcloud-format ply
        """
    )
    
    # Input
    parser.add_argument("--frames", "-f", help="Directory with RGB frames")
    parser.add_argument("--video", "-v", help="Video file for depth computation")
    
    # Output
    parser.add_argument("--output", "-o", default="./depth_output",
                       help="Output directory")
    
    # Model
    parser.add_argument("--model", default="large",
                       choices=["small", "base", "large", "nested-base", "nested-large"],
                       help="Depth Anything 3 model (default: large)")
    
    # Processing
    parser.add_argument("--process-res", type=int, default=None,
                       help="Processing resolution (default: auto=image size)")
    parser.add_argument("--process-method", default="upper",
                       choices=["upper", "lower"],
                       help="Resize method: 'lower' gives higher effective resolution")
    parser.add_argument("--percentiles", type=float, nargs=2, default=[2.0, 98.0],
                       metavar=("LOW", "HIGH"),
                       help="Normalization percentiles (default: 2 98)")
    
    # Point cloud export
    parser.add_argument("--export-pointcloud", action="store_true",
                       help="Export point cloud alongside depth")
    parser.add_argument("--pointcloud-format", default="ply",
                       choices=["ply", "glb"],
                       help="Point cloud format (default: ply)")
    parser.add_argument("--max-points", type=int, default=1_000_000,
                       help="Max points for point cloud (default: 1M)")
    
    # Output format
    parser.add_argument("--no-exr", action="store_true",
                       help="Don't save EXR depth maps")
    parser.add_argument("--no-vis", action="store_true",
                       help="Don't save depth visualizations")
    
    # Performance
    parser.add_argument("--device", default="cuda")
    
    # Debug
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.frames and not args.video:
        print("Error: Must specify --frames or --video")
        return 1
    
    config = DepthPassConfig(
        frames_dir=args.frames or "",
        video_path=args.video or "",
        output_dir=args.output,
        model_size=args.model,
        process_res=args.process_res,
        process_method=args.process_method,
        norm_percentiles=tuple(args.percentiles),
        export_pointcloud=args.export_pointcloud,
        pointcloud_format=args.pointcloud_format,
        max_pointcloud_points=args.max_points,
        save_exr=not args.no_exr,
        save_visualization=not args.no_vis,
        device=args.device,
        verbose=args.verbose,
    )
    
    pipeline = DepthPassPipeline(config)
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
