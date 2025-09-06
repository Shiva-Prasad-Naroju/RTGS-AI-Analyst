"""
Example usage of RTGS AI Analyst for Telangana Hospital Data
"""

from cli import RTGSAIAnalyst
import pandas as pd

def create_sample_hospital_data():
    """Create sample hospital data for demonstration"""
    
    import random
    import numpy as np
    from datetime import datetime, timedelta
    
    # Sample data
    districts = ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Khammam', 'Mahbubnagar']
    hospital_types = ['Government', 'Private', 'Corporate', 'Trust']
    
    data = []
    for i in range(100):
        # Introduce some data quality issues intentionally
        hospital_name = f"Hospital_{i+1}"
        if i % 20 == 0:
            hospital_name = None  # Missing names
            
        district = random.choice(districts)
        h_type = random.choice(hospital_types)
        
        # Bed capacity with some issues
        beds = random.randint(10, 500)
        if i % 30 == 0:
            beds = -beds  # Negative beds (data error)
        if i % 25 == 0:
            beds = None  # Missing bed info
            
        # Establishment year
        est_year = random.randint(1950, 2023)
        est_date = datetime(est_year, random.randint(1, 12), random.randint(1, 28))
        
        # Some inconsistent categorical data
        if random.random() < 0.1:
            h_type = h_type.lower()  # Case inconsistency
        if random.random() < 0.05:
            h_type = h_type + " "  # Extra spaces
            
        data.append({
            'Hospital_ID': f'H{i+1:03d}',
            'Hospital Name': hospital_name,
            'District': district,
            'Type': h_type,
            'Bed_Capacity': beds,
            'Established_Date': est_date,
            'Specialty_Services': random.randint(1, 15),
            'Emergency_Services': random.choice(['Yes', 'No', 'yes', ''])  # Inconsistent
        })
    
    # Add some duplicate rows
    data.extend(data[:5])
    
    df = pd.DataFrame(data)
    return df

def main():
    """Demonstrate RTGS AI Analyst with sample data"""
    
    # Create sample data
    sample_df = create_sample_hospital_data()
    sample_path = "data/raw/sample_hospitals.csv"
    
    # Ensure directory exists
    import os
    os.makedirs("data/raw", exist_ok=True)
    
    # Save sample data
    sample_df.to_csv(sample_path, index=False)
    print(f"Created sample dataset: {sample_path}")
    
    # Run RTGS AI Analyst
    analyst = RTGSAIAnalyst()
    success = analyst.run_full_pipeline(sample_path)
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("Check the outputs/ directory for results")
    else:
        print("\n❌ Analysis failed")

if __name__ == "__main__":
    main()
