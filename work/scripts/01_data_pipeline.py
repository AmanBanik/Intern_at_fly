import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv('../../.env')
HF_TOKEN = os.environ.get('HF_TOKEN')

def run_pipeline():
    print("Running 01_data_pipeline...")
    con = duckdb.connect()
    if HF_TOKEN:
        con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')")
    
    REL = 'hf://datasets/FlyRank/internship-warehouse'
    SAMPLE_TABLE = f"read_parquet('{REL}/fact_content_daily_performance_sample.parquet')"
    
    query = f"""
        WITH bounds AS (
            SELECT MAX(report_date) AS end_d FROM {SAMPLE_TABLE}
        ),
        content_agg AS (
            SELECT 
                f.client_hash_id, 
                f.content_hash_id,
                SUM(CASE WHEN f.report_date BETWEEN b.end_d - INTERVAL 30 DAY AND b.end_d - INTERVAL 16 DAY THEN f.gsc_impressions ELSE 0 END) AS imp_past15,
                AVG(CASE WHEN f.report_date BETWEEN b.end_d - INTERVAL 30 DAY AND b.end_d - INTERVAL 23 DAY THEN f.gsc_avg_position END) AS pos_first_half,
                AVG(CASE WHEN f.report_date BETWEEN b.end_d - INTERVAL 22 DAY AND b.end_d - INTERVAL 16 DAY THEN f.gsc_avg_position END) AS pos_second_half,
                SUM(CASE WHEN f.report_date > b.end_d - INTERVAL 15 DAY THEN f.gsc_impressions ELSE 0 END) AS imp_next15
            FROM {SAMPLE_TABLE} f
            CROSS JOIN bounds b
            GROUP BY 1, 2
            HAVING imp_past15 >= 100
        )
        SELECT 
            *,
            CASE WHEN imp_next15 < (imp_past15 / 1.0) * 0.85 THEN 1 ELSE 0 END AS dropped_traffic_next15d
        FROM content_agg
    """
    
    df = con.sql(query).df()
    os.makedirs('../outputs', exist_ok=True)
    df.to_csv('../outputs/raw_features.csv', index=False)
    print(f"Pipeline complete: generated raw_features.csv with {len(df)} rows.")

if __name__ == '__main__':
    run_pipeline()
