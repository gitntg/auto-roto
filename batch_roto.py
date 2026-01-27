#!/usr/bin/env python
"""
AUTO-ROTO Batch Processor
=========================

Process multiple videos in sequence with the same or different prompts.

USAGE:
    # Process all videos in a folder with same prompt
    python batch_roto.py --input ./videos/ --prompt "person" --output ./output/

    # Process with a config file
    python batch_roto.py --config batch_config.json

CONFIG FILE FORMAT (batch_config.json):
    {
        "default_prompt": "person",
        "default_model": "large",
        "jobs": [
            {
                "input": "video1.mp4",
                "prompt": "person",
                "output": "./output/video1_roto/"
            },
            {
                "input": "video2.mp4",
                "prompt": "car.person",
                "output": "./output/video2_roto/"
            }
        ]
    }
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging


# Environment verification - ensures correct conda environment
REQUIRED_ENV = "autoroto"

def _check_conda_environment():
    """Verify we're running in the correct conda environment."""
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env != REQUIRED_ENV:
        print("\n" + "="*60)
        print("WRONG CONDA ENVIRONMENT")
        print("="*60)
        print(f"\n  Current environment: {current_env or '(none/base)'}")
        print(f"  Required environment: {REQUIRED_ENV}")
        print(f"\n  Please activate the correct environment:")
        print(f"    conda activate {REQUIRED_ENV}")
        print("\n" + "="*60)
        sys.exit(1)

# Run environment check immediately on import
_check_conda_environment()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BatchRoto")


@dataclass
class RotoJob:
    """Single rotoscoping job using SAM3."""
    input_path: str
    output_dir: str
    prompt: Optional[str] = None
    box: Optional[str] = None
    point: Optional[str] = None
    refine_alpha: bool = True
    output_format: str = "exr"
    bit_depth: int = 16

    # Status tracking
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input_path,
            "output": self.output_dir,
            "prompt": self.prompt,
            "box": self.box,
            "point": self.point,
            "refine_alpha": self.refine_alpha,
            "output_format": self.output_format,
            "bit_depth": self.bit_depth,
            "status": self.status,
            "duration": self.duration,
            "error": self.error_message,
        }
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class BatchProcessor:
    """Process multiple videos through AUTO-ROTO."""
    
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.mxf', '.webm'}
    SEQUENCE_EXTENSIONS = {'.exr', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.dpx'}
    
    def __init__(
        self,
        auto_roto_script: str = None,
        python_interpreter: str = None,
        max_workers: int = 1,  # GPU limits parallel processing
    ):
        self.auto_roto_script = auto_roto_script or self._find_script()
        self.python_interpreter = python_interpreter or sys.executable
        self.max_workers = max_workers
        self.jobs: List[RotoJob] = []
    
    def _find_script(self) -> str:
        """Find auto_roto.py script."""
        possible_paths = [
            Path(__file__).parent / "auto_roto.py",
            Path.cwd() / "auto_roto.py",
            Path.home() / "auto_roto" / "auto_roto.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            "Cannot find auto_roto.py. "
            "Set AUTO_ROTO_SCRIPT environment variable or place in current directory."
        )
    
    def add_job(self, job: RotoJob):
        """Add a single job to the queue."""
        self.jobs.append(job)
    
    def add_folder(
        self,
        input_folder: str,
        output_folder: str,
        prompt: str,
        **kwargs
    ):
        """Add all videos in a folder as jobs."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            raise ValueError(f"Input folder not found: {input_folder}")
        
        # Find video files
        videos = []
        for ext in self.VIDEO_EXTENSIONS:
            videos.extend(input_path.glob(f"*{ext}"))
            videos.extend(input_path.glob(f"*{ext.upper()}"))
        
        videos = sorted(set(videos))
        
        if not videos:
            logger.warning(f"No video files found in: {input_folder}")
            return
        
        # Create jobs
        for video in videos:
            output_dir = output_path / f"{video.stem}_roto"
            
            job = RotoJob(
                input_path=str(video),
                output_dir=str(output_dir),
                prompt=prompt,
                **kwargs
            )
            self.add_job(job)
        
        logger.info(f"Added {len(videos)} videos from {input_folder}")
    
    def load_config(self, config_path: str):
        """Load jobs from a config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        defaults = {
            'prompt': config.get('default_prompt'),
            'refine_alpha': config.get('default_refine', True),
            'output_format': config.get('default_format', 'exr'),
            'bit_depth': config.get('default_bit_depth', 16),
        }

        for job_config in config.get('jobs', []):
            # Merge defaults with job config
            job_params = {**defaults, **job_config}

            job = RotoJob(
                input_path=job_params['input'],
                output_dir=job_params.get('output', f"{job_params['input']}_roto"),
                prompt=job_params.get('prompt'),
                box=job_params.get('box'),
                point=job_params.get('point'),
                refine_alpha=job_params.get('refine_alpha', True),
                output_format=job_params.get('output_format', 'exr'),
                bit_depth=job_params.get('bit_depth', 16),
            )
            self.add_job(job)

        logger.info(f"Loaded {len(self.jobs)} jobs from config")
    
    def _build_command(self, job: RotoJob) -> List[str]:
        """Build command line for a job (SAM3, no --sam-model needed)."""
        cmd = [
            self.python_interpreter,
            self.auto_roto_script,
            '--input', job.input_path,
            '--output', job.output_dir,
            '--format', job.output_format,
            '--bit-depth', str(job.bit_depth),
        ]

        if job.prompt:
            cmd.extend(['--prompt', job.prompt])
        elif job.box:
            cmd.extend(['--box', job.box])
        elif job.point:
            cmd.extend(['--point', job.point])
        else:
            raise ValueError(f"Job has no prompt/box/point: {job.input_path}")

        if not job.refine_alpha:
            cmd.append('--no-refine')

        return cmd
    
    def _run_job(self, job: RotoJob) -> RotoJob:
        """Run a single job."""
        job.status = "running"
        job.start_time = time.time()
        
        logger.info(f"Starting: {Path(job.input_path).name}")
        
        try:
            cmd = self._build_command(job)
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if process.returncode == 0:
                job.status = "completed"
                logger.info(f"Completed: {Path(job.input_path).name}")
            else:
                job.status = "failed"
                job.error_message = process.stderr or process.stdout
                logger.error(f"Failed: {Path(job.input_path).name}\n{job.error_message}")
                
        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error_message = "Timeout (2 hours)"
            logger.error(f"Timeout: {Path(job.input_path).name}")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            logger.error(f"Error: {Path(job.input_path).name}\n{e}")
        
        job.end_time = time.time()
        return job
    
    def run(self, parallel: bool = False) -> List[RotoJob]:
        """
        Run all jobs.
        
        Args:
            parallel: Run jobs in parallel (limited by GPU, usually sequential is better)
        
        Returns:
            List of completed jobs
        """
        if not self.jobs:
            logger.warning("No jobs to process")
            return []
        
        logger.info(f"Processing {len(self.jobs)} jobs...")
        
        start_time = time.time()
        
        if parallel and self.max_workers > 1:
            # Parallel processing (use with caution - GPU memory)
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._run_job, job): job for job in self.jobs}
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Job failed: {e}")
        else:
            # Sequential processing (recommended)
            for i, job in enumerate(self.jobs, 1):
                logger.info(f"Job {i}/{len(self.jobs)}")
                self._run_job(job)
        
        total_time = time.time() - start_time
        
        # Summary
        completed = sum(1 for j in self.jobs if j.status == "completed")
        failed = sum(1 for j in self.jobs if j.status == "failed")
        
        logger.info("="*60)
        logger.info(f"Batch Complete: {completed}/{len(self.jobs)} succeeded, {failed} failed")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info("="*60)
        
        return self.jobs
    
    def save_report(self, output_path: str):
        """Save processing report to JSON."""
        report = {
            "total_jobs": len(self.jobs),
            "completed": sum(1 for j in self.jobs if j.status == "completed"),
            "failed": sum(1 for j in self.jobs if j.status == "failed"),
            "jobs": [j.to_dict() for j in self.jobs],
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AUTO-ROTO Batch Processor (SAM3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process all videos in folder
  %(prog)s --input ./videos/ --prompt "person" --output ./output/

  # Process with config file
  %(prog)s --config batch_config.json

  # Process specific videos
  %(prog)s --files video1.mp4 video2.mp4 --prompt "person" --output ./output/

NOTE: Uses SAM3 with built-in text prompting (270k+ concepts).
      No separate GroundingDINO required.
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", help="Input folder with videos")
    input_group.add_argument("--config", "-c", help="JSON config file")
    input_group.add_argument("--files", "-f", nargs='+', help="Specific video files")

    # Common options (no --sam-model, SAM3 has single architecture)
    parser.add_argument("--output", "-o", default="./output", help="Output folder")
    parser.add_argument("--prompt", "-p", help="Text prompt for all videos")
    parser.add_argument("--format", default="exr", choices=["exr", "png", "tiff"])
    parser.add_argument("--no-refine", action="store_true")

    # Processing options
    parser.add_argument("--parallel", action="store_true",
                       help="Run jobs in parallel (use with caution)")
    parser.add_argument("--report", help="Save report to JSON file")

    args = parser.parse_args()

    processor = BatchProcessor()

    if args.config:
        processor.load_config(args.config)

    elif args.input:
        if not args.prompt:
            parser.error("--prompt required when using --input")

        processor.add_folder(
            args.input,
            args.output,
            args.prompt,
            refine_alpha=not args.no_refine,
            output_format=args.format,
        )

    elif args.files:
        if not args.prompt:
            parser.error("--prompt required when using --files")

        for video_file in args.files:
            video_path = Path(video_file)
            output_dir = Path(args.output) / f"{video_path.stem}_roto"

            job = RotoJob(
                input_path=str(video_path),
                output_dir=str(output_dir),
                prompt=args.prompt,
                refine_alpha=not args.no_refine,
                output_format=args.format,
            )
            processor.add_job(job)

    # Run processing
    processor.run(parallel=args.parallel)

    # Save report
    if args.report:
        processor.save_report(args.report)


if __name__ == "__main__":
    main()
