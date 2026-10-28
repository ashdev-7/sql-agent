import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from src.agent.agent import create_analyst_agent
from src.tools.sql_tool import query_sql, list_tables

load_dotenv()
console = Console()


def _has_aws_credentials() -> bool:
    return bool(
        os.getenv("AWS_ACCESS_KEY")
        and os.getenv("AWS_SECRET_KEY")
        and os.getenv("AWS_BUCKET_NAME")
    )


def _local_db_path() -> Path:
    # Same resolution as sql_tool: WAREHOUSE_DB, else next to this file (not cwd).
    env = os.getenv("WAREHOUSE_DB")
    return Path(env) if env else Path(__file__).parent / "data" / "sample" / "warehouse.db"


def seed_local_smartcity_data():
    db_path = _local_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_data AS
        SELECT * FROM (VALUES
            ('v1','VH-001','2024-03-15 08:00:00','37.7749,-122.4194', 75.5, 75.5,
             'South-East','Tesla','Model Y',2024,'Electric', 8, TRUE, '2024-03-15'),
            ('v2','VH-002','2024-03-15 08:01:00','37.7599,-122.4148', 82.3, 82.3,
             'South-East','Toyota','Vios',2022,'Gasoline', 8, FALSE, '2024-03-15'),
            ('v3','VH-003','2024-03-15 09:15:00','37.6213,-122.3790', 64.0, 64.0,
             'South-East','Hyundai','Accent',2023,'Hybrid', 9, FALSE, '2024-03-15'),
            ('v4','VH-001','2024-03-15 10:30:00','37.4419,-122.1430', 95.2, 95.2,
             'South-East','Tesla','Model Y',2024,'Electric', 10, TRUE, '2024-03-15')
        ) t(id, vehicle_id, timestamp, location, speed, speed_kmh,
            direction, make, model, year, fuelType, hour, is_ev, date)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_data AS
        SELECT * FROM (VALUES
            ('w1','VH-001','37.7749,-122.4194','2024-03-15 08:00:00','2024-03-15',
             32.5,'Sunny', 0.0, 12.3, 78, 85.2, 36.4, 'Moderate'),
            ('w2','VH-002','37.7599,-122.4148','2024-03-15 08:01:00','2024-03-15',
             31.8,'Cloudy', 2.1, 15.0, 82, 90.1, 35.9, 'Moderate'),
            ('w3','VH-003','37.6213,-122.3790','2024-03-15 09:15:00','2024-03-15',
             29.0,'Rain', 12.5, 22.0, 90, 110.0, 33.5, 'Unhealthy for Sensitive'),
            ('w4','VH-001','37.4419,-122.1430','2024-03-15 10:30:00','2024-03-15',
             34.2,'Sunny', 0.0, 8.0, 70, 60.5, 38.1, 'Moderate')
        ) t(id, vehicle_id, location, timestamp, date,
            temperature, weatherCondition, precipitation, windSpeed,
            humidity, airQualityIndex, heat_index, aqi_category)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS emergency_data AS
        SELECT * FROM (VALUES
            ('e1','VH-002','inc-1','Accident','2024-03-15 09:30:00',
             '37.6213,-122.3790','Active','Incident on San Francisco - San Jose Expressway',
             TRUE, TRUE, '2024-03-15'),
            ('e2','VH-003','inc-2','None','2024-03-15 10:00:00',
             '37.4419,-122.1430','Resolved','No incident',
             FALSE, FALSE, '2024-03-15'),
            ('e3','VH-001','inc-3','Medical','2024-03-15 11:00:00',
             '37.3382,-121.8863','Resolved','Incident on San Francisco - San Jose Expressway',
             FALSE, TRUE, '2024-03-15')
        ) t(id, vehicle_id, incidentId, type, timestamp,
            location, status, description, is_active, is_real_incident, date)
    """)

    conn.close()
    console.print(
        f"[green]✓ Local SmartCity sample data ready ({db_path})[/green]"
    )


def check_connection_mode() -> bool:
    """Report which backend the agent will use, and seed local data if needed."""
    if _has_aws_credentials():
        console.print(
            "[green]✓ Mode: SmartCity S3 (DuckDB httpfs → Athena tables)[/green]"
        )
        console.print(f"  Bucket: [cyan]{os.getenv('AWS_BUCKET_NAME')}[/cyan]")
        return True

    console.print("[yellow]⚠ Mode: Local DuckDB (no AWS credentials found)[/yellow]")
    console.print(
        "  Set AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET_NAME in .env to use S3 data."
    )
    seed_local_smartcity_data()
    return False


def main():
    console.print(
        Panel.fit(
            "[bold cyan]SmartCity Data Analyst Agent[/bold cyan]\n"
            "Powered by Groq (GPT-OSS 120B) + LangChain + DuckDB/S3\n"
            "Type [yellow]'exit'[/yellow] to quit",
            border_style="cyan",
        )
    )

    check_connection_mode()
    tools = [query_sql, list_tables]
    run_agent = create_analyst_agent(tools)

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            if question.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break
            if not question:
                continue
            response = run_agent(question)
            console.print(f"\n[bold green]Agent:[/bold green] {response}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped.[/yellow]")
            break


if __name__ == "__main__":
    main()
