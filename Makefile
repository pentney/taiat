# Makefile for taiat with Prolog integration

.PHONY: help install develop test clean build-prolog install-gprolog check-gprolog

# Default target
help:
	@echo "Available targets:"
	@echo "  install        - Install taiat package"
	@echo "  develop        - Install in development mode"
	@echo "  test           - Run all tests"
	@echo "  test-prolog    - Run Prolog-specific tests"
	@echo "  build-prolog   - Compile Prolog files"
	@echo "  clean          - Clean build artifacts"
	@echo "  install-gprolog - Install gprolog (Ubuntu/Debian)"
	@echo "  check-gprolog  - Check if gprolog is installed"

# Check if gprolog is available
check-gprolog:
	@echo "Checking for gprolog..."
	@which gplc || (echo "gprolog not found. Run 'make install-gprolog' to install." && exit 1)
	@gplc --version

# Install gprolog (Ubuntu/Debian)
install-gprolog:
	@echo "Installing gprolog..."
	@echo "This will install gprolog using your system's package manager."
	@echo "For other platforms, see setup.py for installation instructions."
	@if command -v apt-get &> /dev/null; then \
		sudo apt-get update && sudo apt-get install -y gprolog; \
	elif command -v dnf &> /dev/null; then \
		sudo dnf install -y gprolog; \
	elif command -v yum &> /dev/null; then \
		sudo yum install -y gprolog; \
	elif command -v pacman &> /dev/null; then \
		sudo pacman -S --noconfirm gprolog; \
	else \
		echo "Unsupported package manager. Please install gprolog manually:"; \
		echo "  Download from: http://www.gprolog.org/"; \
		exit 1; \
	fi
	@echo "gprolog installed successfully"

# Install taiat package
install: check-gprolog
	@echo "Installing taiat package..."
	python setup.py install

# Install in development mode
develop: check-gprolog
	@echo "Installing taiat in development mode..."
	python setup.py develop

# Build Prolog files
build-prolog: check-gprolog
	@echo "Compiling Prolog files..."
	@mkdir -p src/taiat/prolog/compiled
	@for pl_file in src/taiat/prolog/*.pl; do \
		if [ -f "$$pl_file" ]; then \
			base_name=$$(basename "$$pl_file" .pl); \
			echo "Compiling $$pl_file -> src/taiat/prolog/compiled/$$base_name"; \
			gplc "$$pl_file" -o "src/taiat/prolog/compiled/$$base_name"; \
		fi; \
	done

# Run all tests
test:
	@echo "Running all tests..."
	python -m pytest tests/ -v

# Run Prolog-specific tests
test-prolog: build-prolog
	@echo "Running Prolog tests..."
	@cd src/taiat/prolog && python test_path_planner.py
	@echo "Running compiled Prolog tests..."
	@for test_exec in src/taiat/prolog/compiled/test_*; do \
		if [ -f "$$test_exec" ] && [ -x "$$test_exec" ]; then \
			echo "Running $$test_exec"; \
			"$$test_exec"; \
		fi; \
	done

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/taiat/prolog/compiled/
	rm -rf src/taiat.egg-info/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Full development setup
setup-dev: install-gprolog develop build-prolog
	@echo "Development environment setup complete!"

# Quick test (without full compilation)
quick-test:
	@echo "Running quick tests..."
	python -m pytest tests/ -v -x 