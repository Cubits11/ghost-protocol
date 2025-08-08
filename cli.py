#!/usr/bin/env python3
# cli.py - Ghost Protocol Command Line Interface
"""
Ghost Protocol v0.1 - Command Line Interface
Professional CLI for demonstrating emotionally sovereign AI capabilities
"""

import asyncio
import argparse
import json
import sys
import time
from typing import Dict, Any
from pathlib import Path

from ghost_protocol_main import GhostProtocolSystem
from core.constraints import EXAMPLE_CONSTRAINTS


class GhostProtocolCLI:
    """Command-line interface for Ghost Protocol"""

    def __init__(self):
        self.ghost = None
        self.session_history = []

    def print_banner(self):
        """Display Ghost Protocol banner"""
        banner = """
👻 ════════════════════════════════════════════════════════════════════════════════════════════════════════════════

   ██████╗ ██╗  ██╗ ██████╗ ███████╗████████╗    ██████╗ ██████╗  ██████╗ ████████╗ ██████╗  ██████╗ ██████╗ ██╗     
  ██╔════╝ ██║  ██║██╔═══██╗██╔════╝╚══██╔══╝    ██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗██╔════╝██╔═══██╗██║     
  ██║  ███╗███████║██║   ██║███████╗   ██║       ██████╔╝██████╔╝██║   ██║   ██║   ██║   ██║██║     ██║   ██║██║     
  ██║   ██║██╔══██║██║   ██║╚════██║   ██║       ██╔═══╝ ██╔══██╗██║   ██║   ██║   ██║   ██║██║     ██║   ██║██║     
  ╚██████╔╝██║  ██║╚██████╔╝███████║   ██║       ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝╚██████╗╚██████╔╝███████╗
   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝       ╚═╝     ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝

   v0.1 - The first technically enforceable framework for emotionally sovereign AI

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════👻
"""
        print(banner)

    def print_system_status(self):
        """Display current system status"""
        if not self.ghost or not self.ghost.system_initialized:
            print("❌ System not initialized")
            return

        status = self.ghost.get_system_status()

        print("\n🔧 System Status:")
        print(
            f"   Health: {'🟢' if status.get('system_health') == 'healthy' else '🔴'} {status.get('system_health', 'unknown').title()}")

        budget = status.get('privacy_budget', {})
        print(f"   Privacy Budget: {budget.get('remaining_epsilon', 0):.1f}ε / {budget.get('total_epsilon', 8.0)}ε")

        constraints = status.get('constraints', {})
        print(f"   Constraints: {constraints.get('enabled', 0)} active")

        memory = status.get('memory_vault', {})
        print(f"   Memory: {memory.get('total_contexts', 0)} contexts stored")

    def print_help(self):
        """Display available commands"""
        help_text = """
Available Commands:
  help                    - Show this help message
  status                  - Display system status
  demo                    - Run interactive demo mode
  test <message>          - Test single message processing
  examples                - Show example test cases
  constraints             - View active constraints
  history                 - Show session history
  export                  - Export session data
  clear                   - Clear session history
  quit/exit               - Exit Ghost Protocol

Demo Examples:
  test "I'm angry about this!"                    # Emotional boundary test
  test "My email is john@example.com"             # Privacy protection test
  test "Hello, how are you?"                      # Normal interaction test
  test "I'm worried about my SSN 123-45-6789"    # Mixed privacy/emotion test
"""
        print(help_text)

    async def process_message(self, message: str, user_id: str = "cli_user") -> Dict[str, Any]:
        """Process a message through Ghost Protocol"""
        if not self.ghost or not self.ghost.system_initialized:
            return {"error": "System not initialized"}

        try:
            result = await self.ghost.process_user_input(message, user_id)

            # Store in session history
            self.session_history.append({
                "timestamp": time.time(),
                "user_input": message,
                "result": {
                    "response": result.response,
                    "processing_route": result.processing_route,
                    "constraints_applied": result.constraints_applied,
                    "privacy_budget_used": result.privacy_budget_used,
                    "processing_time_ms": result.processing_time_ms,
                    "confidence_score": result.confidence_score
                }
            })

            return result

        except Exception as e:
            error_result = {"error": str(e)}
            self.session_history.append({
                "timestamp": time.time(),
                "user_input": message,
                "result": error_result
            })
            return error_result

    def format_result(self, result) -> str:
        """Format processing result for display"""
        if hasattr(result, 'error'):
            return f"❌ Error: {result.error}"

        output = []
        output.append(f"\n🤖 AI Response: {result.response}")

        # Processing info
        route_emoji = {"blocked": "🚫", "local_only": "💻", "cloud_anonymized": "☁️", "local_default": "🔒"}
        emoji = route_emoji.get(result.processing_route, "❓")
        output.append(f"🔄 Processing Route: {emoji} {result.processing_route}")

        # Constraints
        if result.constraints_applied:
            constraints_str = ", ".join(result.constraints_applied)
            output.append(f"⚖️ Constraints Applied: {constraints_str}")
        else:
            output.append("✅ No Constraint Violations")

        # Metrics
        output.append(f"🔒 Privacy Budget Used: {result.privacy_budget_used}ε")
        output.append(f"⏱️ Processing Time: {result.processing_time_ms:.1f}ms")
        output.append(f"🎯 Confidence: {result.confidence_score:.2f}")

        return "\n".join(output)

    def show_examples(self):
        """Display example test cases"""
        examples = [
            ("Emotional Boundary", "I am so angry and frustrated about this situation!"),
            ("Privacy Protection", "My email is john.doe@example.com and I need help"),
            ("Anxiety Support", "I'm really anxious and worried about my presentation tomorrow"),
            ("Normal Chat", "Hello! How are you doing today?"),
            ("PII Detection", "My phone number is 555-123-4567 and I need assistance"),
            ("Financial Privacy", "My credit card is 4532-1234-5678-9012"),
            ("Health Privacy", "I got diagnosed with anxiety by Dr. Smith at medical@hospital.com"),
            ("Workplace Emotion", "I'm absolutely furious with my boss and want to quit!"),
            ("Mixed Scenario", "I'm worried about my SSN 123-45-6789 being stolen")
        ]

        print("\n📝 Example Test Cases:")
        for i, (category, example) in enumerate(examples, 1):
            print(f"  {i}. {category}: \"{example}\"")

    def show_constraints(self):
        """Display active constraints"""
        print(f"\n📋 Active Constraints:\n")
        print(EXAMPLE_CONSTRAINTS)

    def show_history(self):
        """Display session history"""
        if not self.session_history:
            print("\n📜 No session history")
            return

        print(f"\n📜 Session History ({len(self.session_history)} interactions):")
        for i, entry in enumerate(self.session_history[-10:], 1):  # Show last 10
            result = entry["result"]
            constraints = result.get("constraints_applied", [])
            constraint_str = f" ({', '.join(constraints)})" if constraints else ""

            print(f"  {i}. \"{entry['user_input'][:50]}{'...' if len(entry['user_input']) > 50 else ''}\"")
            print(f"     → {result.get('processing_route', 'unknown')}{constraint_str}")

    def export_session(self):
        """Export session data to JSON"""
        if not self.session_history:
            print("📤 No session data to export")
            return

        filename = f"ghost_protocol_session_{int(time.time())}.json"

        export_data = {
            "session_info": {
                "timestamp": time.time(),
                "total_interactions": len(self.session_history),
                "ghost_protocol_version": "0.1.0"
            },
            "system_status": self.ghost.get_system_status() if self.ghost else {},
            "interactions": self.session_history
        }

        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"📤 Session exported to: {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")

    async def demo_mode(self):
        """Interactive demo mode"""
        print("\n🎭 Interactive Demo Mode")
        print("Type 'help' for commands, 'quit' to exit\n")

        while True:
            try:
                user_input = input("👻 > ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("👻 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.print_help()
                elif user_input.lower() == 'status':
                    self.print_system_status()
                elif user_input.lower() == 'examples':
                    self.show_examples()
                elif user_input.lower() == 'constraints':
                    self.show_constraints()
                elif user_input.lower() == 'history':
                    self.show_history()
                elif user_input.lower() == 'export':
                    self.export_session()
                elif user_input.lower() == 'clear':
                    self.session_history.clear()
                    print("🧹 Session history cleared")
                elif user_input.lower().startswith('test '):
                    message = user_input[5:]  # Remove 'test '
                    print(f"\n📤 Processing: \"{message}\"")
                    result = await self.process_message(message)
                    print(self.format_result(result))
                else:
                    # Treat as direct message
                    print(f"\n📤 Processing: \"{user_input}\"")
                    result = await self.process_message(user_input)
                    print(self.format_result(result))

            except KeyboardInterrupt:
                print("\n\n👻 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    async def run_single_test(self, message: str):
        """Run a single test message"""
        print(f"📤 Testing: \"{message}\"")
        result = await self.process_message(message)
        print(self.format_result(result))
        self.print_system_status()

    async def run_benchmark(self):
        """Run performance benchmark"""
        test_cases = [
            "I'm angry about this!",
            "My email is test@example.com",
            "Hello there!",
            "I'm worried about my SSN 123-45-6789",
            "Help me with something normal"
        ]

        print("\n🏃‍♂️ Running Performance Benchmark...")
        total_time = 0

        for i, test_case in enumerate(test_cases, 1):
            start_time = time.time()
            result = await self.process_message(test_case, f"bench_user_{i}")
            end_time = time.time()

            processing_time = (end_time - start_time) * 1000
            total_time += processing_time

            print(f"  {i}. \"{test_case[:30]}{'...' if len(test_case) > 30 else ''}\"")
            print(f"     ⏱️ {processing_time:.1f}ms - {result.processing_route}")

        avg_time = total_time / len(test_cases)
        print(f"\n📊 Benchmark Results:")
        print(f"   Average Processing Time: {avg_time:.1f}ms")
        print(f"   Total Tests: {len(test_cases)}")
        print(f"   All Tests Completed Successfully: {'✅' if avg_time < 500 else '⚠️'}")

    async def initialize_system(self):
        """Initialize Ghost Protocol system"""
        print("🔄 Initializing Ghost Protocol...")
        try:
            self.ghost = GhostProtocolSystem()
            if self.ghost.system_initialized:
                print("✅ Ghost Protocol initialized successfully!")
                self.print_system_status()
                return True
            else:
                print("❌ Ghost Protocol initialization failed!")
                return False
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Ghost Protocol v0.1 - Emotionally Sovereign AI CLI")
    parser.add_argument('--demo', action='store_true', help='Run interactive demo mode')
    parser.add_argument('--test', type=str, help='Test a single message')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--examples', action='store_true', help='Show example test cases')
    parser.add_argument('--constraints', action='store_true', help='Show active constraints')
    parser.add_argument('--no-banner', action='store_true', help='Skip banner display')

    args = parser.parse_args()

    cli = GhostProtocolCLI()

    if not args.no_banner:
        cli.print_banner()

    # Initialize system
    if not await cli.initialize_system():
        sys.exit(1)

    try:
        if args.demo:
            await cli.demo_mode()
        elif args.test:
            await cli.run_single_test(args.test)
        elif args.benchmark:
            await cli.run_benchmark()
        elif args.status:
            cli.print_system_status()
        elif args.examples:
            cli.show_examples()
        elif args.constraints:
            cli.show_constraints()
        else:
            # Default to demo mode if no specific command
            print("💡 No command specified, starting interactive demo mode...")
            await cli.demo_mode()

    except KeyboardInterrupt:
        print("\n\n👻 Goodbye!")
    finally:
        if cli.ghost:
            cli.ghost.close()


if __name__ == "__main__":
    asyncio.run(main())