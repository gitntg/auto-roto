#!/usr/bin/env python
"""
AUTO-ROTO Nuke Integration
==========================

This module provides Nuke integration for the AUTO-ROTO pipeline.
It can be imported in Nuke's Script Editor or menu.py.

INSTALLATION:
    1. Copy this file to your ~/.nuke/ directory
    2. Add to menu.py:
        import auto_roto_nuke
        auto_roto_nuke.add_to_menu()

USAGE:
    In Nuke:
    - Select a Read node with video/sequence
    - Run: Nodes > AUTO-ROTO > Process Selected

Or via Python:
    import auto_roto_nuke
    auto_roto_nuke.process_selected()
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Check if running inside Nuke
try:
    import nuke
    import nukescripts
    IN_NUKE = True
except ImportError:
    IN_NUKE = False
    print("Warning: Not running inside Nuke. Some features disabled.")


# Path to auto_roto.py script - update this to your installation path
AUTO_ROTO_SCRIPT = os.environ.get(
    'AUTO_ROTO_SCRIPT',
    os.path.expanduser('~/auto_roto/auto_roto.py')
)

# Default Python interpreter (should have all dependencies)
PYTHON_INTERPRETER = os.environ.get(
    'AUTO_ROTO_PYTHON',
    sys.executable  # Use Nuke's Python by default
)


def get_frame_range(node):
    """Get frame range from a Nuke node."""
    first = int(node['first'].value())
    last = int(node['last'].value())
    return first, last


def get_file_path(node):
    """Get file path from a Read node."""
    return node['file'].value()


def create_roto_read(output_dir: str, prefix: str = "roto", first: int = 1, last: int = 100):
    """Create a Read node pointing to the roto output."""
    if not IN_NUKE:
        print(f"Would create Read node: {output_dir}/rgba/{prefix}.####.exr")
        return None
    
    # Construct file path
    file_path = os.path.join(output_dir, "rgba", f"{prefix}.%04d.exr")
    
    # Create Read node
    read_node = nuke.createNode('Read')
    read_node['file'].setValue(file_path)
    read_node['first'].setValue(first)
    read_node['last'].setValue(last)
    read_node['origfirst'].setValue(first)
    read_node['origlast'].setValue(last)
    read_node['label'].setValue('AUTO-ROTO')
    read_node['tile_color'].setValue(0x7f00ffff)
    
    return read_node


def process_node(
    source_node,
    prompt: str = None,
    box: str = None,
    output_dir: str = None,
    sam_model: str = "large",
    refine_alpha: bool = True,
    output_format: str = "exr",
    callback=None
):
    """
    Process a Nuke node through AUTO-ROTO.
    
    Args:
        source_node: Nuke Read node or node with 'file' knob
        prompt: Text prompt for object detection
        box: Box coordinates "x1,y1,x2,y2"
        output_dir: Output directory (auto-generated if None)
        sam_model: SAM2 model size
        refine_alpha: Enable alpha refinement
        output_format: Output format (exr, png, tiff)
        callback: Progress callback function(progress: float, message: str)
    
    Returns:
        Path to output directory
    """
    if not IN_NUKE:
        raise RuntimeError("Must run inside Nuke")
    
    # Get source file
    source_file = get_file_path(source_node)
    if not source_file:
        raise ValueError("Selected node has no file path")
    
    # Get frame range
    first, last = get_frame_range(source_node)
    
    # Generate output directory if not provided
    if output_dir is None:
        source_name = os.path.splitext(os.path.basename(source_file))[0]
        # Remove frame padding
        source_name = source_name.replace('%04d', '').replace('####', '').strip('.')
        output_dir = os.path.join(
            os.path.dirname(source_file),
            f"{source_name}_roto"
        )
    
    # Build command
    cmd = [
        PYTHON_INTERPRETER,
        AUTO_ROTO_SCRIPT,
        '--input', source_file,
        '--output', output_dir,
        '--sam-model', sam_model,
        '--format', output_format,
    ]
    
    if prompt:
        cmd.extend(['--prompt', prompt])
    elif box:
        cmd.extend(['--box', box])
    else:
        raise ValueError("Must provide either prompt or box")
    
    if not refine_alpha:
        cmd.append('--no-refine')
    
    # Run process
    if callback:
        callback(0.0, "Starting AUTO-ROTO...")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Monitor progress
    for line in process.stdout:
        print(line, end='')  # Print to Nuke's console
        
        # Parse progress from output
        if 'Frame' in line and '/' in line:
            try:
                parts = line.split('Frame')[1].split('/')
                current = int(parts[0].strip())
                total = int(parts[1].split()[0].strip())
                progress = current / total
                if callback:
                    callback(progress, f"Processing frame {current}/{total}")
            except:
                pass
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"AUTO-ROTO failed with code {process.returncode}")
    
    if callback:
        callback(1.0, "Complete!")
    
    return output_dir


def process_selected(prompt: str = None):
    """
    Process the selected node through AUTO-ROTO.
    
    Shows a dialog if prompt is not provided.
    """
    if not IN_NUKE:
        print("Must run inside Nuke")
        return
    
    # Get selected node
    selected = nuke.selectedNodes()
    if not selected:
        nuke.message("Please select a Read node first")
        return
    
    node = selected[0]
    if node.Class() != 'Read':
        nuke.message("Please select a Read node")
        return
    
    # Get prompt via panel if not provided
    if prompt is None:
        panel = nuke.Panel("AUTO-ROTO")
        panel.addSingleLineInput("Text Prompt", "person")
        panel.addEnumerationPulldown("Model", "large small base_plus tiny")
        panel.addBooleanCheckBox("Refine Alpha", True)
        panel.addEnumerationPulldown("Format", "exr png tiff")
        
        if not panel.show():
            return
        
        prompt = panel.value("Text Prompt")
        sam_model = panel.value("Model")
        refine = panel.value("Refine Alpha")
        fmt = panel.value("Format")
    else:
        sam_model = "large"
        refine = True
        fmt = "exr"
    
    # Process with progress dialog
    task = nuke.ProgressTask("AUTO-ROTO")
    
    def progress_callback(progress, message):
        task.setProgress(int(progress * 100))
        task.setMessage(message)
    
    try:
        output_dir = process_node(
            node,
            prompt=prompt,
            sam_model=sam_model,
            refine_alpha=refine,
            output_format=fmt,
            callback=progress_callback
        )
        
        # Create output Read node
        first, last = get_frame_range(node)
        roto_node = create_roto_read(output_dir, "roto", first, last)
        
        # Position below source
        roto_node.setXpos(node.xpos())
        roto_node.setYpos(node.ypos() + 100)
        
        nuke.message(f"AUTO-ROTO complete!\nOutput: {output_dir}")
        
    except Exception as e:
        nuke.message(f"AUTO-ROTO failed:\n{str(e)}")
    
    finally:
        del task


def add_to_menu():
    """Add AUTO-ROTO to Nuke's menu."""
    if not IN_NUKE:
        print("Must run inside Nuke")
        return
    
    # Find or create menu
    menu = nuke.menu('Nodes')
    roto_menu = menu.addMenu('AUTO-ROTO', icon='AutoPaint.png')
    
    # Add commands
    roto_menu.addCommand(
        'Process Selected',
        'auto_roto_nuke.process_selected()',
        'ctrl+shift+r'
    )
    
    roto_menu.addCommand(
        'Process Selected (Person)',
        'auto_roto_nuke.process_selected("person")'
    )
    
    roto_menu.addCommand(
        'Process Selected (Custom Prompt)...',
        'auto_roto_nuke.process_selected()'
    )
    
    print("AUTO-ROTO menu added to Nuke")


# ============================================================================
# Batch Node Setup
# ============================================================================

def setup_batch_roto(folder_path: str, prompt: str = "person"):
    """
    Set up batch processing for a folder of videos.
    
    Creates a Nuke script structure for processing multiple shots.
    """
    if not IN_NUKE:
        print("Must run inside Nuke")
        return
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Find all video files
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mxf'}
    videos = [f for f in folder.iterdir() if f.suffix.lower() in video_extensions]
    
    if not videos:
        nuke.message("No video files found in folder")
        return
    
    # Create nodes for each video
    y_pos = 0
    for i, video in enumerate(videos):
        # Create Read node
        read = nuke.createNode('Read')
        read['file'].setValue(str(video))
        read['label'].setValue(video.stem)
        read.setXpos(0)
        read.setYpos(y_pos)
        
        # Store prompt in node metadata
        tab = nuke.Tab_Knob('autoroto_tab', 'AUTO-ROTO')
        prompt_knob = nuke.String_Knob('autoroto_prompt', 'Prompt')
        prompt_knob.setValue(prompt)
        status_knob = nuke.Text_Knob('autoroto_status', 'Status', 'Pending')
        
        read.addKnob(tab)
        read.addKnob(prompt_knob)
        read.addKnob(status_knob)
        
        y_pos += 150
    
    nuke.message(f"Added {len(videos)} shots for batch processing.\n"
                 f"Select nodes and run 'Process Selected' to roto each shot.")


# ============================================================================
# CLI for testing outside Nuke
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AUTO-ROTO Nuke Integration Test")
    parser.add_argument("--test", action="store_true", help="Run integration test")
    
    args = parser.parse_args()
    
    if args.test:
        print("AUTO-ROTO Nuke Integration")
        print(f"  Script path: {AUTO_ROTO_SCRIPT}")
        print(f"  Python: {PYTHON_INTERPRETER}")
        print(f"  Running in Nuke: {IN_NUKE}")
        
        if os.path.exists(AUTO_ROTO_SCRIPT):
            print("  ✓ auto_roto.py found")
        else:
            print("  ✗ auto_roto.py NOT FOUND")
