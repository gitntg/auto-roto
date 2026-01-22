#!/bin/bash
# ============================================================================
# AUTO-ROTO Installation Script
# ============================================================================
# 
# This script installs all dependencies for production-grade auto-rotoscoping:
#   - PyTorch with CUDA support
#   - SAM2 (Segment Anything Model 2.1)
#   - GroundingDINO (optional, for text prompts)
#   - OpenEXR (for 16/32-bit output)
#   - All Python dependencies
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# For minimal install (SAM2 only):
#   ./install.sh --minimal
#
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Parse arguments
MINIMAL=false
for arg in "$@"; do
    case $arg in
        --minimal)
            MINIMAL=true
            shift
            ;;
    esac
done

print_header "AUTO-ROTO Installation"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi
print_success "Python $PYTHON_VERSION"

# Check CUDA
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    print_success "NVIDIA driver: $CUDA_VERSION"
else
    print_warning "CUDA not detected, will use CPU (much slower)"
fi

# Create virtual environment (optional)
read -p "Create new virtual environment? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header "Creating Virtual Environment"
    python -m venv autoroto_env
    # Windows uses Scripts/, Unix uses bin/
    if [ -f "autoroto_env/Scripts/activate" ]; then
        source autoroto_env/Scripts/activate
    else
        source autoroto_env/bin/activate
    fi
    print_success "Virtual environment created and activated"
fi

# ============================================================================
# INSTALL PYTORCH
# ============================================================================
print_header "Installing PyTorch"

# Detect CUDA version and install appropriate PyTorch
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision
fi

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
print_success "PyTorch installed"

# ============================================================================
# INSTALL CORE DEPENDENCIES
# ============================================================================
print_header "Installing Core Dependencies"

pip install numpy opencv-python pillow
pip install matplotlib scipy
pip install tqdm requests

print_success "Core dependencies installed"

# ============================================================================
# INSTALL SAM2
# ============================================================================
print_header "Installing SAM2 (Segment Anything Model 2)"

# Check if already installed
if python -c "import sam2" 2>/dev/null; then
    print_warning "SAM2 already installed"
else
    # Clone and install
    if [ ! -d "sam2" ]; then
        git clone https://github.com/facebookresearch/sam2.git
    fi
    cd sam2
    pip install -e ".[notebooks]"
    cd ..
    print_success "SAM2 installed"
fi

# Download SAM2 checkpoints
print_header "Downloading SAM2 Model Checkpoints"

CHECKPOINT_DIR="$HOME/.cache/sam2"
mkdir -p "$CHECKPOINT_DIR"

download_checkpoint() {
    local name=$1
    local url=$2
    local path="$CHECKPOINT_DIR/$name"
    
    if [ -f "$path" ]; then
        print_warning "$name already exists"
    else
        echo "Downloading $name..."
        wget -q --show-progress -O "$path" "$url"
        print_success "$name downloaded"
    fi
}

# Download all SAM2.1 checkpoints
download_checkpoint "sam2.1_hiera_tiny.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
download_checkpoint "sam2.1_hiera_small.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
download_checkpoint "sam2.1_hiera_base_plus.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
download_checkpoint "sam2.1_hiera_large.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

# ============================================================================
# INSTALL GROUNDINGDINO (optional)
# ============================================================================
if [ "$MINIMAL" = false ]; then
    print_header "Installing GroundingDINO (Text-Based Detection)"
    
    if python -c "import groundingdino" 2>/dev/null; then
        print_warning "GroundingDINO already installed"
    else
        if [ ! -d "GroundingDINO" ]; then
            git clone https://github.com/IDEA-Research/GroundingDINO.git
        fi
        cd GroundingDINO
        pip install -e .
        cd ..
        print_success "GroundingDINO installed"
    fi
    
    # Download checkpoint
    GDINO_DIR="$HOME/.cache/groundingdino"
    mkdir -p "$GDINO_DIR"
    
    if [ ! -f "$GDINO_DIR/groundingdino_swint_ogc.pth" ]; then
        echo "Downloading GroundingDINO checkpoint..."
        wget -q --show-progress -O "$GDINO_DIR/groundingdino_swint_ogc.pth" \
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        print_success "GroundingDINO checkpoint downloaded"
    else
        print_warning "GroundingDINO checkpoint already exists"
    fi
fi

# ============================================================================
# INSTALL DEPTH ANYTHING V2
# ============================================================================
if [ "$MINIMAL" = false ]; then
    print_header "Installing Depth Anything V2"
    
    if python -c "from transformers import AutoModelForDepthEstimation" 2>/dev/null; then
        print_warning "Transformers already installed (can load Depth Anything V2)"
    else
        pip install transformers
        print_success "Transformers installed"
    fi
    
    # Option 1: Clone official repo (recommended for direct usage)
    if [ ! -d "Depth-Anything-V2" ]; then
        git clone https://github.com/DepthAnything/Depth-Anything-V2.git
        print_success "Depth Anything V2 repo cloned"
    else
        print_warning "Depth Anything V2 repo already exists"
    fi
    
    # Download checkpoints
    DEPTH_DIR="$HOME/.cache/depth_anything_v2"
    mkdir -p "$DEPTH_DIR"
    
    download_depth_checkpoint() {
        local name=$1
        local url=$2
        local path="$DEPTH_DIR/$name"
        
        if [ -f "$path" ]; then
            print_warning "$name already exists"
        else
            echo "Downloading $name..."
            wget -q --show-progress -O "$path" "$url"
            print_success "$name downloaded"
        fi
    }
    
    # Download Depth Anything V2 checkpoints
    download_depth_checkpoint "depth_anything_v2_vits.pth" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    download_depth_checkpoint "depth_anything_v2_vitb.pth" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
    download_depth_checkpoint "depth_anything_v2_vitl.pth" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
fi

# ============================================================================
# INSTALL OPENEXR
# ============================================================================
print_header "Installing OpenEXR Support"

# Try to install OpenEXR Python bindings
if pip install OpenEXR 2>/dev/null; then
    print_success "OpenEXR installed via pip"
else
    print_warning "OpenEXR pip install failed, trying alternative..."
    
    # Install system dependencies (for building OpenEXR)
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y libopenexr-dev openexr
        pip install OpenEXR
    elif command -v brew &> /dev/null; then
        brew install openexr
        pip install OpenEXR
    else
        print_warning "Could not install OpenEXR. EXR output will use OpenCV fallback."
    fi
fi

# ============================================================================
# FINAL VERIFICATION
# ============================================================================
print_header "Verifying Installation"

echo "Testing imports..."

python << 'EOF'
import sys

def test_import(name, package=None):
    try:
        if package:
            exec(f"import {package}")
        else:
            exec(f"import {name}")
        print(f"  ✓ {name}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: {e}")
        return False

print("\nCore packages:")
test_import("torch")
test_import("torchvision")
test_import("numpy")
test_import("cv2", "cv2")
test_import("PIL", "PIL")

print("\nSAM2:")
test_import("sam2")

print("\nOptional:")
test_import("groundingdino")
test_import("OpenEXR")

print("\nDepth Anything V2:")
try:
    from transformers import AutoModelForDepthEstimation
    print("  ✓ Can load via Transformers")
except ImportError:
    print("  ✗ Transformers not available")

# Check for cached checkpoints
import os
cache_dir = os.path.expanduser("~/.cache/depth_anything_v2")
if os.path.exists(cache_dir):
    checkpoints = os.listdir(cache_dir)
    if checkpoints:
        print(f"  ✓ Checkpoints cached: {', '.join(checkpoints)}")
    else:
        print("  ⚠ No checkpoints downloaded")
else:
    print("  ⚠ Checkpoint cache not found")

print("\nGPU Status:")
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  ⚠ CUDA not available, using CPU")
EOF

print_header "Installation Complete!"

echo -e "${GREEN}AUTO-ROTO is ready to use!${NC}"
echo ""
echo "Quick start (SAM2 only):"
echo "  python auto_roto.py --input video.mp4 --prompt \"person\" --output ./output"
echo ""
echo "Full pipeline (SAM2 + Depth Refinement):"
echo "  python full_pipeline.py --input video.mp4 --prompt \"person\" --output ./output"
echo ""
echo "Depth refinement only (on existing alpha mattes):"
echo "  python depth_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./refined"
echo ""
echo "For help:"
echo "  python auto_roto.py --help"
echo "  python depth_refine.py --help"
echo "  python full_pipeline.py --help"
echo ""
