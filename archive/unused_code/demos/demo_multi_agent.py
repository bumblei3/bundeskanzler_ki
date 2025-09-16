"""
Multi-Agent System Demo & Test Script
====================================

Comprehensive demonstration and testing of the Multi-Agent Intelligence System
for Bundeskanzler-KI with specialized Politik, Wirtschaft, and Klima agents.

Author: Advanced RAG Team
Date: September 15, 2025
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))

from core.advanced_rag_system import AdvancedRAGSystem
from core.multi_agent_system import AgentType, MultiAgentSystem, QueryComplexity


class MultiAgentDemo:
    """Demonstration class for Multi-Agent System capabilities"""

    def __init__(self):
        self.system = None

    async def initialize(self):
        """Initialize the multi-agent system"""
        print("🚀 Initializing Multi-Agent Intelligence System...")
        print("=" * 60)

        try:
            # Initialize RAG system first
            print("📚 Loading Advanced RAG System...")
            rag_system = AdvancedRAGSystem()

            # Initialize Multi-Agent System
            print("🤖 Initializing Multi-Agent Coordinator...")
            self.system = MultiAgentSystem(rag_system)

            print("✅ Multi-Agent System ready!")
            print()

            return True

        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            return False

    def display_system_info(self):
        """Display system information and capabilities"""
        print("🔍 SYSTEM INFORMATION")
        print("=" * 60)

        # Get agent capabilities
        capabilities = self.system.get_agent_capabilities()

        for agent_name, info in capabilities.items():
            emoji = {"politik": "🏛️", "wirtschaft": "💰", "klima": "🌍"}.get(agent_name, "🤖")

            print(f"{emoji} {agent_name.upper()} AGENT:")
            print(f"   Domain: {info['domain']}")
            print(f"   Expertise Areas: {len(info['expertise_areas'])} specializations")
            print(f"   Keywords: {info['keyword_count']} domain-specific terms")
            print(f"   Confidence Threshold: {info['confidence_threshold']:.1%}")
            print()

        # System status
        status = self.system.get_status()
        print("📊 SYSTEM STATUS:")
        print(f"   Total Agents: {status['system_info']['total_agents']}")
        print(f"   RAG System: {status['system_info']['rag_system_type']}")
        print(
            f"   Classification Domains: {', '.join(status['system_info']['classification_domains'])}"
        )
        print()

    async def run_comprehensive_tests(self):
        """Run comprehensive test suite"""
        print("🧪 COMPREHENSIVE MULTI-AGENT TESTS")
        print("=" * 60)

        # Test cases for different domains and complexities
        test_cases = [
            # Politik Tests
            {
                "category": "Politik",
                "query": "Wer ist der aktuelle Bundeskanzler und welche Partei regiert?",
                "expected_agent": "politik",
                "complexity": "simple",
            },
            {
                "category": "Politik",
                "query": "Wie funktioniert die Gesetzgebung im Bundestag?",
                "expected_agent": "politik",
                "complexity": "moderate",
            },
            # Wirtschaft Tests
            {
                "category": "Wirtschaft",
                "query": "Wie hoch ist die aktuelle Inflationsrate in Deutschland?",
                "expected_agent": "wirtschaft",
                "complexity": "simple",
            },
            {
                "category": "Wirtschaft",
                "query": "Welche Auswirkungen hat die Zinspolitik der EZB auf den deutschen Arbeitsmarkt?",
                "expected_agent": "wirtschaft",
                "complexity": "moderate",
            },
            # Klima Tests
            {
                "category": "Klima",
                "query": "Was sind die deutschen CO2-Ziele bis 2030?",
                "expected_agent": "klima",
                "complexity": "simple",
            },
            {
                "category": "Klima",
                "query": "Wie steht es um den Fortschritt der Energiewende in Deutschland?",
                "expected_agent": "klima",
                "complexity": "moderate",
            },
            # Multi-Domain Tests
            {
                "category": "Multi-Domain",
                "query": "Wie beeinflusst die Klimapolitik die deutsche Wirtschaft und welche politischen Maßnahmen sind geplant?",
                "expected_agent": "multi",
                "complexity": "complex",
            },
            {
                "category": "Multi-Domain",
                "query": "Welche wirtschaftlichen Auswirkungen hat der Kohleausstieg auf die Arbeitsplätze?",
                "expected_agent": "multi",
                "complexity": "complex",
            },
        ]

        results = []
        total_start_time = time.time()

        for i, test_case in enumerate(test_cases, 1):
            print(f"🔬 TEST {i}: {test_case['category']}")
            print(f"   Query: {test_case['query']}")

            start_time = time.time()

            try:
                response = await self.system.query(test_case["query"])
                processing_time = time.time() - start_time

                # Analyze response
                agent_count = len(response.contributing_agents)
                primary_agent = (
                    response.contributing_agents[0].value
                    if response.contributing_agents
                    else "none"
                )

                print(f"   ✅ Response received in {processing_time:.3f}s")
                print(
                    f"   🤖 Contributing Agents: {agent_count} ({[a.value for a in response.contributing_agents]})"
                )
                print(f"   📊 Confidence: {response.confidence:.1%}")
                print(f"   📚 Sources: {len(response.sources)}")
                print(f"   📝 Response Length: {len(response.primary_response)} chars")
                print()

                # Store results
                results.append(
                    {
                        "test_id": i,
                        "category": test_case["category"],
                        "query": test_case["query"],
                        "expected_agent": test_case["expected_agent"],
                        "actual_agent": primary_agent,
                        "agent_count": agent_count,
                        "confidence": response.confidence,
                        "processing_time": processing_time,
                        "sources_count": len(response.sources),
                        "response_length": len(response.primary_response),
                        "success": True,
                    }
                )

            except Exception as e:
                processing_time = time.time() - start_time
                print(f"   ❌ Test failed: {str(e)}")
                print()

                results.append(
                    {
                        "test_id": i,
                        "category": test_case["category"],
                        "query": test_case["query"],
                        "expected_agent": test_case["expected_agent"],
                        "actual_agent": "error",
                        "agent_count": 0,
                        "confidence": 0.0,
                        "processing_time": processing_time,
                        "sources_count": 0,
                        "response_length": 0,
                        "success": False,
                        "error": str(e),
                    }
                )

        total_time = time.time() - total_start_time

        # Display test summary
        self.display_test_summary(results, total_time)

        return results

    def display_test_summary(self, results: List[Dict], total_time: float):
        """Display comprehensive test summary"""
        print("📊 TEST SUMMARY")
        print("=" * 60)

        successful_tests = [r for r in results if r["success"]]
        failed_tests = [r for r in results if not r["success"]]

        print(
            f"✅ Successful Tests: {len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results)*100:.1f}%)"
        )
        print(f"❌ Failed Tests: {len(failed_tests)}")
        print(f"⏱️ Total Test Time: {total_time:.2f}s")
        print()

        if successful_tests:
            avg_response_time = sum(r["processing_time"] for r in successful_tests) / len(
                successful_tests
            )
            avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
            avg_sources = sum(r["sources_count"] for r in successful_tests) / len(successful_tests)

            print("📈 PERFORMANCE METRICS:")
            print(f"   Average Response Time: {avg_response_time:.3f}s")
            print(f"   Average Confidence: {avg_confidence:.1%}")
            print(f"   Average Sources per Response: {avg_sources:.1f}")
            print()

            # Agent usage statistics
            agent_usage = {}
            for result in successful_tests:
                agent = result["actual_agent"]
                agent_usage[agent] = agent_usage.get(agent, 0) + 1

            print("🤖 AGENT USAGE:")
            for agent, count in agent_usage.items():
                percentage = count / len(successful_tests) * 100
                emoji = {"politik": "🏛️", "wirtschaft": "💰", "klima": "🌍"}.get(agent, "🤖")
                print(f"   {emoji} {agent.title()}: {count} queries ({percentage:.1f}%)")
            print()

            # Category performance
            categories = {}
            for result in successful_tests:
                cat = result["category"]
                if cat not in categories:
                    categories[cat] = {"count": 0, "total_time": 0, "total_confidence": 0}
                categories[cat]["count"] += 1
                categories[cat]["total_time"] += result["processing_time"]
                categories[cat]["total_confidence"] += result["confidence"]

            print("📊 CATEGORY PERFORMANCE:")
            for cat, stats in categories.items():
                avg_time = stats["total_time"] / stats["count"]
                avg_conf = stats["total_confidence"] / stats["count"]
                print(f"   {cat}: {avg_time:.3f}s avg, {avg_conf:.1%} confidence")
            print()

        if failed_tests:
            print("❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   Test {test['test_id']}: {test['error']}")
            print()

    async def interactive_demo(self):
        """Interactive demonstration mode"""
        print("🎮 INTERACTIVE MULTI-AGENT DEMO")
        print("=" * 60)
        print("Enter your questions to test the Multi-Agent System!")
        print("Examples:")
        print("  • 'Wer ist der Bundeskanzler?' (Politik)")
        print("  • 'Wie hoch ist die Inflation?' (Wirtschaft)")
        print("  • 'Was sind die CO2-Ziele?' (Klima)")
        print("  • 'Wie beeinflusst Klimapolitik die Wirtschaft?' (Multi-Domain)")
        print("\nType 'quit' to exit, 'status' for system status")
        print("-" * 60)

        while True:
            try:
                query = input("\n🤖 Your Question: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if query.lower() == "status":
                    status = self.system.get_status()
                    print("\n📊 SYSTEM STATUS:")
                    print(json.dumps(status, indent=2, default=str))
                    continue

                if not query:
                    continue

                print(f"\n🔍 Processing: '{query}'")
                start_time = time.time()

                response = await self.system.query(query)
                processing_time = time.time() - start_time

                print(f"\n{'='*60}")
                print("🤖 MULTI-AGENT RESPONSE:")
                print(f"{'='*60}")
                print(response.primary_response)
                print(f"\n{'='*60}")
                print(f"📊 Metadata:")
                print(
                    f"   🤖 Contributing Agents: {[a.value for a in response.contributing_agents]}"
                )
                print(f"   📊 Confidence: {response.confidence:.1%}")
                print(f"   ⏱️ Processing Time: {processing_time:.3f}s")
                print(f"   📚 Sources: {len(response.sources)}")

                if response.sources:
                    print(f"\n📚 Top Sources:")
                    for i, source in enumerate(response.sources[:3], 1):
                        preview = (
                            source["text"][:100] + "..."
                            if len(source["text"]) > 100
                            else source["text"]
                        )
                        print(f"   {i}. {preview}")

            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


async def main():
    """Main demo execution"""
    print("🚀 MULTI-AGENT INTELLIGENCE SYSTEM DEMO")
    print("=" * 60)
    print("Advanced Bundeskanzler-KI with specialized Politik, Wirtschaft & Klima agents")
    print("Date: September 15, 2025")
    print("=" * 60)
    print()

    demo = MultiAgentDemo()

    # Initialize system
    if not await demo.initialize():
        print("❌ Failed to initialize system. Exiting.")
        return

    # Display system information
    demo.display_system_info()

    # Menu
    while True:
        print("🎯 SELECT DEMO MODE:")
        print("1. 🧪 Run Comprehensive Tests")
        print("2. 🎮 Interactive Demo")
        print("3. 🔍 System Status")
        print("4. 🚪 Exit")

        choice = input("\nYour choice (1-4): ").strip()

        if choice == "1":
            print()
            await demo.run_comprehensive_tests()
            input("\nPress Enter to continue...")
            print()

        elif choice == "2":
            print()
            await demo.interactive_demo()
            print()

        elif choice == "3":
            print()
            status = demo.system.get_status()
            print("📊 DETAILED SYSTEM STATUS:")
            print("=" * 60)
            print(json.dumps(status, indent=2, default=str))
            input("\nPress Enter to continue...")
            print()

        elif choice == "4":
            print("👋 Exiting Multi-Agent Demo. Thank you!")
            break

        else:
            print("❌ Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback

        traceback.print_exc()
