# Makefile for Ghost Protocol v0.1
# Professional development and deployment automation

.PHONY: help install test lint format clean demo build deploy benchmark docs

# Default target
help: ## Show this help message
	@echo "👻 Ghost Protocol v0.1 - Development Commands"
	@echo "════════════════════════════════════════════"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development
install: ## Install development dependencies
	@echo "📦 Installing Ghost Protocol dependencies..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

install-dev: ## Install development dependencies
	@echo "🛠️ Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest black flake8 mypy pytest-benchmark pytest-asyncio
	@echo "✅ Development environment ready!"

# Testing
test: ## Run all tests
	@echo "🧪 Running Ghost Protocol tests..."
	python -m pytest tests/ -v --tb=short
	@echo "✅ All tests passed!"

test-integration: ## Run integration tests
	@echo "🔗 Running integration tests..."
	python -m pytest tests/test_integration.py -v
	@echo "✅ Integration tests passed!"

test-coverage: ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	python -m pytest tests/ --cov=core --cov=ghost_protocol_main --cov-report=html --cov-report=term
	@echo "📋 Coverage report generated in htmlcov/"

# Code Quality
lint: ## Run linting checks
	@echo "🔍 Running code quality checks..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	@echo "✅ Linting complete!"

format: ## Format code with black
	@echo "🎨 Formatting code..."
	black . --line-length=88
	@echo "✅ Code formatted!"

type-check: ## Run type checking
	@echo "🔍 Running type checks..."
	mypy core/ ghost_protocol_main.py cli.py --ignore-missing-imports
	@echo "✅ Type checking complete!"

quality: format lint type-check ## Run all code quality checks
	@echo "✨ All quality checks complete!"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "🏃‍♂️ Running Ghost Protocol benchmarks..."
	python cli.py --benchmark
	@echo "📊 Benchmark complete!"

benchmark-detailed: ## Run detailed performance analysis
	@echo "🔬 Running detailed benchmarks..."
	python -m pytest benchmarks/ -v --benchmark-only --benchmark-sort=mean
	@echo "📈 Detailed benchmark complete!"

# Demo and Testing
demo: ## Start interactive demo
	@echo "🎭 Starting Ghost Protocol demo..."
	python cli.py --demo

demo-web: ## Start Streamlit web demo
	@echo "🌐 Starting web demo..."
	streamlit run ui/demo.py --server.port=8501 --server.headless=false

test-examples: ## Test all example scenarios
	@echo "📝 Testing example scenarios..."
	python cli.py --test "I'm angry about this!"
	python cli.py --test "My email is john@example.com"
	python cli.py --test "Hello, how are you?"
	@echo "✅ Example tests complete!"

# Docker Operations
build: ## Build Docker image
	@echo "🐳 Building Ghost Protocol Docker image..."
	docker build -t ghost-protocol:0.1.0 .
	@echo "✅ Docker image built!"

build-prod: ## Build production Docker image
	@echo "🏭 Building production Docker image..."
	docker build -t ghost-protocol:0.1.0 --target production .
	@echo "✅ Production image built!"

deploy: ## Deploy with Docker Compose
	@echo "🚀 Deploying Ghost Protocol..."
	docker-compose up --build -d
	@echo "✅ Deployment complete! Access at http://localhost:8501"

deploy-full: ## Deploy with monitoring
	@echo "🚀 Deploying Ghost Protocol with monitoring..."
	docker-compose --profile monitoring up --build -d
	@echo "✅ Full deployment complete!"

stop: ## Stop Docker deployment
	@echo "🛑 Stopping Ghost Protocol..."
	docker-compose down
	@echo "✅ Deployment stopped!"

logs: ## View deployment logs
	@echo "📜 Viewing Ghost Protocol logs..."
	docker-compose logs -f ghost-protocol

# Database and Data
db-backup: ## Backup encrypted database
	@echo "💾 Backing up Ghost Protocol database..."
	mkdir -p backups
	cp data/databases/*.db backups/ghost_backup_$(shell date +%Y%m%d_%H%M%S).db 2>/dev/null || echo "No database found"
	@echo "✅ Database backup complete!"

db-clean: ## Clean database (WARNING: destroys all data)
	@echo "🧹 Cleaning Ghost Protocol database..."
	@read -p "Are you sure? This will delete all emotional data [y/N]: " confirm && [ "$$confirm" = "y" ]
	rm -f data/databases/*.db
	rm -f ghost_protocol.db
	rm -f ghost_memory.db
	@echo "✅ Database cleaned!"

# Documentation
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	mkdir -p docs/api
	python -c "
import core
help(core.constraints)
help(core.vault)
" > docs/api/core_api.txt
	@echo "✅ Documentation generated!"

docs-serve: ## Serve documentation locally
	@echo "📖 Serving documentation..."
	python -m http.server 8080 --directory docs
	@echo "📋 Documentation available at http://localhost:8080"

# Cleanup
clean: ## Clean temporary files
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleanup complete!"

clean-all: clean ## Clean everything including data
	@echo "🧹 Deep cleaning..."
	rm -rf data/logs/*
	rm -rf .venv
	docker system prune -f
	@echo "✅ Deep cleanup complete!"

# Release and Packaging
package: ## Create distribution package
	@echo "📦 Creating distribution package..."
	python setup.py sdist bdist_wheel
	@echo "✅ Package created in dist/"

release-check: test lint type-check benchmark ## Pre-release validation
	@echo "🔍 Running pre-release checks..."
	python ghost_protocol_main.py
	python cli.py --test "Hello world"
	@echo "✅ Pre-release validation complete!"

# Development Workflows
dev-setup: install-dev ## Complete development setup
	@echo "🛠️ Setting up development environment..."
	pre-commit install 2>/dev/null || echo "pre-commit not available"
	@echo "✅ Development environment ready!"

quick-test: ## Quick test for development
	@echo "⚡ Running quick tests..."
	python -c "from ghost_protocol_main import GhostProtocolSystem; g = GhostProtocolSystem(); print('✅ System OK')"
	python cli.py --test "Hello" --no-banner
	@echo "✅ Quick test complete!"

# CI/CD Helpers
ci-test: ## Run tests in CI environment
	@echo "🤖 Running CI tests..."
	python -m pytest tests/ -v --tb=short --junitxml=test-results.xml
	@echo "✅ CI tests complete!"

ci-build: ## Build for CI/CD
	@echo "🤖 Building for CI/CD..."
	docker build --target production -t ghost-protocol:latest .
	@echo "✅ CI build complete!"

# Monitoring and Health
health-check: ## Check system health
	@echo "🏥 Checking Ghost Protocol health..."
	python -c "
from ghost_protocol_main import GhostProtocolSystem
import json
ghost = GhostProtocolSystem()
status = ghost.get_system_status()
print('System Health:', status.get('system_health', 'unknown'))
print('Components:', status.get('constraints', {}).get('enabled', 0), 'constraints loaded')
print('Privacy Budget:', status.get('privacy_budget', {}).get('remaining_epsilon', 0), 'ε available')
ghost.close()
"
	@echo "✅ Health check complete!"

performance-profile: ## Profile system performance
	@echo "📊 Profiling Ghost Protocol performance..."
	python -m cProfile -o profile.stats cli.py --test "Performance test"
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
	@echo "✅ Performance profiling complete!"

# Help and Information
version: ## Show version information
	@echo "👻 Ghost Protocol v0.1.0"
	@echo "📅 Built: $(shell date)"
	@echo "🐍 Python: $(shell python --version)"
	@echo "🐳 Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"

info: ## Show system information
	@echo "👻 Ghost Protocol v0.1 - System Information"
	@echo "════════════════════════════════════════════"
	@echo "📂 Project Directory: $(PWD)"
	@echo "🐍 Python Version: $(shell python --version)"
	@echo "📦 Dependencies: $(shell pip list | wc -l) packages installed"
	@echo "🐳 Docker: $(shell docker --version 2>/dev/null || echo 'Not available')"
	@echo "💾 Database: $(shell ls *.db 2>/dev/null | wc -l) files found"
	@echo "📊 Tests: $(shell find tests/ -name "*.py" | wc -l) test files"
	@echo "📁 Components: $(shell find core/ -name "*.py" | wc -l) core modules"

# All-in-one commands
all: clean install test lint build ## Run complete build pipeline
	@echo "✅ Complete build pipeline finished!"

demo-full: build deploy demo-web ## Complete demo setup
	@echo "🎭 Full demo environment ready!"