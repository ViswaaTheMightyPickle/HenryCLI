#!/bin/bash
# HenryCLI Installation Script for Mac/Linux
# This script installs HenryCLI and its dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
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

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check system requirements
print_header "HenryCLI Installation"
echo ""

# Check Python version
print_header "Checking Python"
if check_command python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python 3 found: $PYTHON_VERSION"
    
    # Check if Python >= 3.10
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        print_error "Python 3.10 or higher is required"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    echo "Please install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

# Check pip
print_header "Checking pip"
if check_command pip3; then
    print_success "pip3 found"
elif check_command pip; then
    print_success "pip found"
    PIP_CMD="pip"
else
    print_warning "pip not found, will attempt to install"
fi

PIP_CMD="${PIP_CMD:-pip3}"

# Check for LM Studio
print_header "Checking LM Studio"
if check_command lms; then
    print_success "LM Studio CLI (lms) found"
    LMS_VERSION=$(lms --version 2>/dev/null || echo "unknown")
    echo "  Version: $LMS_VERSION"
else
    print_warning "LM Studio CLI not found"
    echo "  HenryCLI can work with LM Studio server API"
    echo "  Install LM Studio from: https://lmstudio.ai/"
fi

# Check if LM Studio server is running
print_header "Checking LM Studio Server"
if curl -s http://localhost:1234/health &> /dev/null; then
    print_success "LM Studio server is running on port 1234"
else
    print_warning "LM Studio server is not running"
    echo "  Please start LM Studio and enable the server"
fi

# Create virtual environment (optional)
print_header "Installation Options"
echo ""
echo "Where would you like to install HenryCLI?"
echo "  1) System-wide (requires sudo)"
echo "  2) User directory (recommended)"
echo "  3) Virtual environment"
echo ""
read -p "Choose option [1-3]: " INSTALL_OPTION

case $INSTALL_OPTION in
    1)
        print_header "Installing System-wide"
        sudo $PIP_CMD install -e ".[dev]"
        print_success "HenryCLI installed system-wide"
        ;;
    2)
        print_header "Installing to User Directory"
        $PIP_CMD install --user -e ".[dev]"
        print_success "HenryCLI installed to user directory"
        
        # Add to PATH if needed
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            USER_BIN="$HOME/.local/bin"
            if [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
                print_warning "Adding $USER_BIN to PATH"
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
                if [ -f ~/.zshrc ]; then
                    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
                fi
                echo "Please run: source ~/.bashrc (or restart terminal)"
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            USER_BIN="$HOME/Library/Python/${PYTHON_MAJOR}.${PYTHON_MINOR}/bin"
            if [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
                print_warning "Adding $USER_BIN to PATH"
                echo 'export PATH="$HOME/Library/Python/${PYTHON_MAJOR}.${PYTHON_MINOR}/bin:$PATH"' >> ~/.zshrc
                echo "Please run: source ~/.zshrc (or restart terminal)"
            fi
        fi
        ;;
    3)
        print_header "Creating Virtual Environment"
        VENV_DIR="${HENRYCLI_VENV:-$HOME/.henrycli-venv}"
        
        if [ -d "$VENV_DIR" ]; then
            print_warning "Virtual environment already exists at $VENV_DIR"
            read -p "Remove and recreate? [y/N]: " RECREATE
            if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
                rm -rf "$VENV_DIR"
            else
                print_error "Installation cancelled"
                exit 1
            fi
        fi
        
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        $PIP_CMD install -e ".[dev]"
        
        print_success "HenryCLI installed in virtual environment"
        echo "  Location: $VENV_DIR"
        echo "  To activate: source $VENV_DIR/bin/activate"
        
        # Add activation to shell config
        read -p "Add auto-activation to shell config? [y/N]: " AUTO_ACTIVATE
        if [ "$AUTO_ACTIVATE" = "y" ] || [ "$AUTO_ACTIVATE" = "Y" ]; then
            echo "source $VENV_DIR/bin/activate" >> ~/.bashrc
            if [ -f ~/.zshrc ]; then
                echo "source $VENV_DIR/bin/activate" >> ~/.zshrc
            fi
            print_success "Added auto-activation to shell config"
        fi
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

# Verify installation
print_header "Verifying Installation"
if command -v henry &> /dev/null; then
    HENRY_VERSION=$(henry version)
    print_success "HenryCLI installed: $HENRY_VERSION"
else
    print_warning "henry command not found in PATH"
    echo "  You may need to restart your terminal or add it to PATH"
fi

# Create config directory
print_header "Setting Up Configuration"
CONFIG_DIR="$HOME/.henrycli"
if [ ! -d "$CONFIG_DIR" ]; then
    mkdir -p "$CONFIG_DIR/models"
    mkdir -p "$CONFIG_DIR/contexts/active"
    mkdir -p "$CONFIG_DIR/contexts/completed"
    mkdir -p "$CONFIG_DIR/filestore"
    print_success "Created configuration directory: $CONFIG_DIR"
else
    print_success "Configuration directory already exists"
fi

# Install recommended plugins
print_header "LM Studio Plugins (Optional)"
echo ""
echo "Recommended LM Studio plugins:"
echo "  - DuckDuckGo Search: lms get danielsig/duckduckgo"
echo "  - Visit Website: lms get danielsig/visit-website"
echo "  - BigRAG (document search): lms get picklerick/big-rag-rust-accelerated"
echo ""
read -p "Install recommended plugins? [y/N]: " INSTALL_PLUGINS

if [ "$INSTALL_PLUGINS" = "y" ] || [ "$INSTALL_PLUGINS" = "Y" ]; then
    if check_command lms; then
        print_success "Installing DuckDuckGo plugin..."
        lms get danielsig/duckduckgo || print_warning "Failed to install DuckDuckGo"
        
        print_success "Installing Visit Website plugin..."
        lms get danielsig/visit-website || print_warning "Failed to install Visit Website"
        
        print_success "Installing BigRAG plugin..."
        lms get picklerick/big-rag-rust-accelerated || print_warning "Failed to install BigRAG"
        
        print_success "Plugins installed!"
    else
        print_warning "LM Studio CLI not available, skipping plugin installation"
    fi
fi

# Final summary
print_header "Installation Complete!"
echo ""
echo -e "${GREEN}HenryCLI has been installed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Start LM Studio and enable the server (port 1234)"
echo "  2. Download a model (e.g., Phi-3-mini-4k-instruct)"
echo "  3. Run: henry health"
echo "  4. Try: henry analyze \"Write a hello world program\""
echo ""
echo "Documentation: https://github.com/ViswaaTheMightyPickle/HenryCLI"
echo ""

# Show help
echo "Quick reference:"
echo "  henry version     - Show version"
echo "  henry health      - Check LM Studio connection"
echo "  henry analyze     - Analyze a task"
echo "  henry run         - Execute a task"
echo "  henry models      - Manage models"
echo "  henry config      - View/edit configuration"
echo "  henry --help      - Show all commands"
