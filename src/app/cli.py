import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import typer
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.provider import get_chat_model, get_embedding_model, require_env_for, choose_provider
from tools.search_adapter import require_tavily_api_key
from tools.embeddings import embed_texts
from graph.builder import build_graph
from graph.state import GraphState

app = typer.Typer(help="Skill Gap Analyst CLI - Live Analysis Only")


def ensure_directory(path: Path) -> None:
    """Ensure directory exists, create if needed."""
    path.mkdir(parents=True, exist_ok=True)


def extract_source_stats(final_state: GraphState) -> Dict[str, any]:
    """Extract source usage statistics from final state."""
    stats = {
        "total_docs_fetched": 0,
        "top_sources": [],
        "sources_by_component": {}
    }
    
    # Market summary sources
    if final_state.market_summary and hasattr(final_state.market_summary, 'sources_sample'):
        market_sources = final_state.market_summary.sources_sample or []
        stats["sources_by_component"]["market_intelligence"] = len(market_sources)
        stats["total_docs_fetched"] += len(market_sources)
        
        # Add to top sources
        for source in market_sources[:3]:  # Top 3 from market
            stats["top_sources"].append(f"Market: {source}")
    
    # Extract URLs from report if available (learning resources)
    if final_state.report_md:
        import re
        # Extract URLs from markdown links
        url_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
        urls = re.findall(url_pattern, final_state.report_md)
        
        learning_resources = len(urls)
        stats["sources_by_component"]["learning_resources"] = learning_resources
        stats["total_docs_fetched"] += learning_resources
        
        # Add top learning resource URLs
        for title, url in urls[:2]:  # Top 2 learning resources
            stats["top_sources"].append(f"Learning: {url}")
    
    # Limit top sources to 5
    stats["top_sources"] = stats["top_sources"][:5]
    
    return stats


def save_artifact(data: dict, filepath: Path) -> None:
    """Save data as JSON artifact with proper serialization."""
    ensure_directory(filepath.parent)
    
    # Convert any Pydantic models to dicts recursively
    def convert_to_serializable(obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_data = convert_to_serializable(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


@app.command()
def analyze(
    cv: str = typer.Option(..., "--cv", help="Path to CV file"),
    role: str = typer.Option("Senior AI Engineer", "--role", help="Target role to analyze"),
    region: str = typer.Option("Global", "--region", help="Region for market analysis"),
    lang: str = typer.Option("en", "--lang", help="Language for output"),
    provider: str = typer.Option("auto", "--provider", help="LLM provider: auto|gemini|mistral (default: auto)"),
    outdir: str = typer.Option("outputs", "--outdir", help="Output directory"),
    allow_offline_embeddings: bool = typer.Option(
        False, 
        "--allow-offline-embeddings/--no-allow-offline-embeddings", 
        help="Allow offline embedding fallback (default: False, requires live embeddings)"
    ),
):
    """Analyze skill gaps for a given CV and role using live APIs only."""
    
    # Load environment variables
    load_dotenv()
    
    try:
        typer.echo("🚀 Starting Live Skill Gap Analysis...")
        typer.echo("   🔒 Live APIs Only - No Mock Data")
        typer.echo(f"   CV: {cv}")
        typer.echo(f"   Target Role: {role}")
        typer.echo(f"   Region: {region}")
        typer.echo(f"   Language: {lang}")
        typer.echo(f"   Provider: {provider}")
        typer.echo(f"   Offline Embeddings: {'Allowed' if allow_offline_embeddings else 'Disabled'}")
        
        # Validate API keys based on provider selection
        typer.echo("🔑 Validating API keys...")
        try:
            # Check LLM provider keys based on selection
            if provider == "gemini":
                require_env_for("gemini")
                typer.echo("   ✅ Gemini API key validated")
            elif provider == "mistral":
                require_env_for("mistral")
                typer.echo("   ✅ Mistral API key validated")
            elif provider == "auto":
                # Auto mode: check for available API keys
                has_gemini = bool(os.getenv("GEMINI_API_KEY"))
                has_mistral = bool(os.getenv("MISTRAL_API_KEY"))
                
                if not has_gemini and not has_mistral:
                    raise RuntimeError("At least one API key required for auto mode: GEMINI_API_KEY or MISTRAL_API_KEY")
                elif not has_gemini:
                    typer.echo("   ⚠️  GEMINI_API_KEY missing - will use Mistral only")
                elif not has_mistral:
                    typer.echo("   ⚠️  MISTRAL_API_KEY missing - will use Gemini only")
                else:
                    typer.echo("   ✅ Auto mode: Both Gemini and Mistral keys available")
            else:
                raise RuntimeError(f"Unsupported provider: {provider}. Use: auto, gemini, or mistral")
            
            # Check Tavily API key
            require_tavily_api_key()
            typer.echo("   ✅ Tavily API key validated")
            
        except RuntimeError as e:
            typer.echo(f"   ❌ API Key validation failed: {str(e)}", err=True)
            typer.echo("", err=True)
            typer.echo("💡 Required API keys:", err=True)
            if provider == "gemini":
                typer.echo("   • GEMINI_API_KEY (for Gemini provider)", err=True)
            elif provider == "mistral":
                typer.echo("   • MISTRAL_API_KEY (for Mistral provider)", err=True)
            else:  # auto
                typer.echo("   • GEMINI_API_KEY and/or MISTRAL_API_KEY (at least one required for auto)", err=True)
            typer.echo("   • TAVILY_API_KEY (for web search)", err=True)
            typer.echo("", err=True)
            typer.echo("Set these in your .env file or environment variables.", err=True)
            raise typer.Exit(1)
        
        # Set embedding configuration
        if allow_offline_embeddings:
            os.environ['ALLOW_OFFLINE_EMBEDDINGS'] = 'true'
            typer.echo("   ⚠️  Offline embedding fallback enabled")
        else:
            os.environ['ALLOW_OFFLINE_EMBEDDINGS'] = 'false'
            typer.echo("   🔒 Live embeddings required (no fallbacks)")
        
        # Validate CV file exists
        cv_path = Path(cv)
        if not cv_path.exists():
            typer.echo(f"❌ Error: CV file not found: {cv}", err=True)
            raise typer.Exit(1)
        
        # Setup output directories
        output_dir = Path(outdir)
        artifacts_dir = output_dir / "artifacts"
        ensure_directory(output_dir)
        ensure_directory(artifacts_dir)
        
        typer.echo(f"📁 Output directory: {output_dir.absolute()}")
        
        # Initialize LLM provider
        typer.echo("🧠 Initializing live LLM provider...")
        typer.echo(f"   Selected provider: {provider}")
        
        try:
            # Get models using new provider system
            chat_model = get_chat_model(provider, None)
            embedding_model = get_embedding_model(provider, None)
            
            # Determine which provider was actually used (for Auto mode)
            actual_provider = "unknown"
            try:
                if hasattr(chat_model, 'model'):
                    model_name = chat_model.model
                    if 'gemini' in model_name.lower():
                        actual_provider = "Gemini"
                    elif 'mistral' in model_name.lower():
                        actual_provider = "Mistral"
            except:
                pass
            
            # Log provider details
            typer.echo("   ✅ LLM provider initialized successfully")
            
            # Show actual provider used for Auto mode
            if provider == "auto" and actual_provider != "unknown":
                typer.echo(f"   � Auto mode resolved to: {actual_provider}")
            
            # Try to get model names
            chat_model_name = "Unknown"
            embed_model_name = "Unknown"
            
            try:
                if hasattr(chat_model, 'model'):
                    chat_model_name = chat_model.model
                elif hasattr(chat_model, 'model_name'):
                    chat_model_name = chat_model.model_name
            except:
                pass
                
            try:
                if hasattr(embedding_model, 'model'):
                    embed_model_name = embedding_model.model
                elif hasattr(embedding_model, 'model_name'):
                    embed_model_name = embedding_model.model_name
            except:
                pass
            
            typer.echo(f"   📡 Chat model: {chat_model_name}")
            typer.echo(f"   🔢 Embedding model: {embed_model_name}")
            typer.echo("   🚫 Mock implementations disabled")
            
        except Exception as e:
            typer.echo(f"   ❌ Error initializing LLM provider: {str(e)}", err=True)
            raise typer.Exit(1)
        
        # Build the analysis graph
        typer.echo("🔗 Building analysis pipeline...")
        try:
            graph = build_graph(provider_choice=provider)
            typer.echo("✅ Analysis pipeline built successfully")
        except Exception as e:
            typer.echo(f"❌ Error building pipeline: {str(e)}", err=True)
            raise typer.Exit(1)
        
        # Create initial state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"skill_gap_report_{timestamp}.md"
        report_path = output_dir / report_filename
        
        initial_state = GraphState(
            file_path=str(cv_path.absolute()),
            target_role=role,
            market_region=region,
            lang=lang,
            output_path=str(report_path.absolute()),
            logs=[],
            provider=provider,
            chat_model_name=chat_model_name,
            embed_model_name=embed_model_name
        )
        
        typer.echo("⚡ Executing analysis pipeline...")
        
        # Execute the graph
        try:
            # Execute graph and collect final state
            result = graph.invoke(initial_state.model_dump())
            final_state = GraphState(**result)
            
            # Save intermediate artifacts
            typer.echo("💾 Saving analysis artifacts...")
            
            # Save CV structure if available
            if final_state.cv_struct:
                cv_artifact_path = artifacts_dir / f"cv_structure_{timestamp}.json"
                save_artifact(final_state.cv_struct, cv_artifact_path)
                typer.echo(f"   📄 CV structure: {cv_artifact_path}")
            
            # Save skill profile if available
            if final_state.skill_profile:
                skill_artifact_path = artifacts_dir / f"skill_profile_{timestamp}.json"
                save_artifact(final_state.skill_profile, skill_artifact_path)
                typer.echo(f"   🎯 Skill profile: {skill_artifact_path}")
            
            # Save market summary if available
            if final_state.market_summary:
                market_artifact_path = artifacts_dir / f"market_summary_{timestamp}.json"
                save_artifact(final_state.market_summary, market_artifact_path)
                typer.echo(f"   📊 Market summary: {market_artifact_path}")
            
            # Save final report
            if final_state.report_md:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(final_state.report_md)
                typer.echo(f"   📋 Final report: {report_path}")
            
            # Save execution logs
            logs_path = artifacts_dir / f"execution_logs_{timestamp}.json"
            save_artifact({"logs": final_state.logs, "timestamp": timestamp}, logs_path)
            typer.echo(f"   📜 Execution logs: {logs_path}")
            
            # Extract and display source usage statistics
            source_stats = extract_source_stats(final_state)
            
            typer.echo("")
            typer.echo("🎉 Analysis completed successfully!")
            typer.echo("")
            typer.echo("📋 Analysis Summary:")
            typer.echo(f"   • CV analyzed: {cv_path.name}")
            typer.echo(f"   • Target role: {role}")
            typer.echo(f"   • Market region: {region}")
            if final_state.report_md:
                typer.echo(f"   • Report size: {len(final_state.report_md):,} characters")
            
            # Display live data usage
            typer.echo("")
            typer.echo("📊 Live Data Usage:")
            typer.echo(f"   • Total docs fetched: {source_stats['total_docs_fetched']}")
            
            if source_stats['sources_by_component']:
                typer.echo("   • Sources by component:")
                for component, count in source_stats['sources_by_component'].items():
                    typer.echo(f"     - {component}: {count} docs")
            
            if source_stats['top_sources']:
                typer.echo("")
                typer.echo("🔗 Top 5 Source URLs Used:")
                for i, source in enumerate(source_stats['top_sources'], 1):
                    typer.echo(f"   {i}. {source}")
            
            typer.echo("")
            typer.echo("📁 Output files:")
            typer.echo(f"   • Report: {report_path}")
            typer.echo(f"   • Artifacts: {artifacts_dir}")
            
            # Show any errors if present
            if final_state.error:
                typer.echo("")
                typer.echo("⚠️  Note: Some errors occurred during analysis:")
                typer.echo(f"   {final_state.error}")
            
        except Exception as e:
            typer.echo(f"❌ Error during analysis execution: {str(e)}", err=True)
            # Save error logs for debugging
            error_logs_path = artifacts_dir / f"error_logs_{timestamp}.json"
            save_artifact({"error": str(e), "timestamp": timestamp}, error_logs_path)
            typer.echo(f"📜 Error logs saved: {error_logs_path}")
            raise typer.Exit(1)
            
    except KeyboardInterrupt:
        typer.echo("\n⏹️  Analysis interrupted by user", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Unexpected error: {str(e)}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
