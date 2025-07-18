#!/bin/bash

# Build script for Taiat Haskell Path Planner

set -e

echo "Building Taiat Haskell Path Planner..."

# Check if cabal is installed
if ! command -v cabal &> /dev/null; then
    echo "Error: cabal is not installed. Please install Haskell Platform or GHC with cabal."
    echo "Visit: https://www.haskell.org/platform/"
    exit 1
fi

# Check if GHC is installed
if ! command -v ghc &> /dev/null; then
    echo "Error: GHC is not installed. Please install Haskell Platform or GHC."
    echo "Visit: https://www.haskell.org/platform/"
    exit 1
fi

echo "GHC version: $(ghc --version)"
echo "Cabal version: $(cabal --version)"

# Clean previous builds
echo "Cleaning previous builds..."
cabal clean

# Update cabal index
echo "Updating cabal index..."
cabal update

# Build the project
echo "Building project..."
cabal build

# Copy the binary to the current directory
echo "Copying binary..."
BINARY_PATH=$(find dist-newstyle -name "taiat-path-planner" -type f | head -n 1)

if [ -n "$BINARY_PATH" ]; then
    cp "$BINARY_PATH" ./taiat-path-planner
    chmod +x ./taiat-path-planner
    echo "Binary copied to: ./taiat-path-planner"
else
    echo "Error: Could not find compiled binary"
    exit 1
fi

# Test the binary
echo "Testing binary..."
if ./taiat-path-planner --version &> /dev/null; then
    echo "Binary test successful!"
else
    echo "Warning: Binary test failed, but build completed"
fi

echo "Build completed successfully!"
echo "You can now run: ./taiat-path-planner" 