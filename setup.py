#!/usr/bin/env python3
"""
Setup script for taiat with Prolog integration.
Handles compilation of Prolog files during installation.
"""

import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


class PrologInstallCommand(install):
    """Custom install command that compiles Prolog files."""
    
    def run(self):
        # Run the normal install
        install.run(self)
        # Compile Prolog files
        self.compile_prolog_files()
    
    def compile_prolog_files(self):
        """Compile Prolog files to executables."""
        prolog_dir = os.path.join(self.install_lib, 'taiat', 'prolog')
        compiled_dir = os.path.join(prolog_dir, 'compiled')
        
        # Create compiled directory
        os.makedirs(compiled_dir, exist_ok=True)
        
        # Find all .pl files
        pl_files = []
        for root, dirs, files in os.walk(prolog_dir):
            for file in files:
                if file.endswith('.pl'):
                    pl_files.append(os.path.join(root, file))
        
        # Compile each .pl file
        for pl_file in pl_files:
            try:
                base_name = os.path.splitext(os.path.basename(pl_file))[0]
                output_file = os.path.join(compiled_dir, base_name)
                
                # Compile with gprolog
                cmd = ['gplc', pl_file, '-o', output_file]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"Compiled {pl_file} -> {output_file}")
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to compile {pl_file}: {e}")
            except FileNotFoundError:
                print("Warning: gprolog not found. Prolog files will not be compiled.")
                break


class PrologDevelopCommand(develop):
    """Custom develop command that compiles Prolog files."""
    
    def run(self):
        # Run the normal develop
        develop.run(self)
        # Compile Prolog files
        self.compile_prolog_files()
    
    def compile_prolog_files(self):
        """Compile Prolog files to executables."""
        prolog_dir = os.path.join('src', 'taiat', 'prolog')
        compiled_dir = os.path.join(prolog_dir, 'compiled')
        
        # Create compiled directory
        os.makedirs(compiled_dir, exist_ok=True)
        
        # Find all .pl files
        pl_files = []
        for root, dirs, files in os.walk(prolog_dir):
            for file in files:
                if file.endswith('.pl'):
                    pl_files.append(os.path.join(root, file))
        
        # Compile each .pl file
        for pl_file in pl_files:
            try:
                base_name = os.path.splitext(os.path.basename(pl_file))[0]
                output_file = os.path.join(compiled_dir, base_name)
                
                # Compile with gprolog
                cmd = ['gplc', pl_file, '-o', output_file]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"Compiled {pl_file} -> {output_file}")
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to compile {pl_file}: {e}")
            except FileNotFoundError:
                print("Warning: gprolog not found. Prolog files will not be compiled.")
                break


def check_gprolog():
    """Check if gprolog is available."""
    try:
        subprocess.run(['gplc', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_gprolog_install_instructions():
    """Get platform-specific gprolog installation instructions."""
    import platform
    
    system = platform.system().lower()
    
    if system == "linux":
        # Try to detect the distribution
        try:
            with open('/etc/os-release', 'r') as f:
                content = f.read().lower()
                if 'ubuntu' in content or 'debian' in content:
                    return "sudo apt-get update && sudo apt-get install gprolog"
                elif 'fedora' in content:
                    return "sudo dnf install gprolog"
                elif 'centos' in content or 'rhel' in content:
                    return "sudo yum install gprolog"
                elif 'arch' in content:
                    return "sudo pacman -S gprolog"
        except FileNotFoundError:
            pass
        return "sudo apt-get install gprolog  # or equivalent for your distribution"
    
    elif system == "darwin":
        return "brew install gprolog"
    
    elif system == "windows":
        return "Download from http://www.gprolog.org/ and add to PATH"
    
    else:
        return "Download from http://www.gprolog.org/"


if __name__ == "__main__":
    # Check for gprolog
    if not check_gprolog():
        print("=" * 60)
        print("WARNING: gprolog not found!")
        print("=" * 60)
        print("taiat includes Prolog modules that require gprolog for full functionality.")
        print("Without gprolog, Prolog features will be limited.")
        print()
        print("To install gprolog, run:")
        print(f"  {get_gprolog_install_instructions()}")
        print()
        print("After installing gprolog, reinstall taiat to compile Prolog files:")
        print("  pip install --force-reinstall .")
        print("=" * 60)
        print()
    
    setup(
        cmdclass={
            'install': PrologInstallCommand,
            'develop': PrologDevelopCommand,
        },
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        include_package_data=True,
        package_data={
            'taiat.prolog': ['*.pl', 'compiled/*'],
        },
    ) 