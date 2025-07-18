#!/bin/bash
# Build script for taiat-path-planner using minimal Cabal setup

set -e

echo "Building taiat-path-planner with minimal Cabal setup..."

# Check if cabal is available
if ! command -v cabal &> /dev/null; then
    echo "Error: cabal not found. Please install cabal-install first."
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
cabal clean

# Build the binary
echo "Building binary..."
cabal build

# Find and copy the binary to current directory
echo "Copying binary to current directory..."
BINARY_PATH=$(find dist-newstyle -name "taiat-path-planner" -type f | head -1)
if [ -n "$BINARY_PATH" ]; then
    cp "$BINARY_PATH" .
    chmod +x taiat-path-planner
    echo "✅ Build successful! Binary created: ./taiat-path-planner"
else
    echo "❌ Binary not found in build output!"
    exit 1
fi

echo "Build completed successfully!" 