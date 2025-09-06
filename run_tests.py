"""
Basic testing script for RTGS AI Analyst
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from agents.ingestion_agent import DataIngestionAgent
from agents.inspector_agent import InspectorAgent
from agents.cleaning_agent import CleaningAgent
from agents.transforming_agent import TransformingAgent
from agents.verification_agent import VerificationAgent
from agents.analysis_agent import AnalysisAgent

def create_test_data():
    """Create test dataset with known issues"""
    data = {
        'Hospital_ID': ['H001', 'H002', 'H003', 'H004', 'H005', 'H001'],  # Duplicate
        'Hospital Name': ['City Hospital', None, 'Rural Clinic', 'Metro Hospital', 'District Hospital', 'City Hospital'],
        'District': ['Hyderabad', 'Warangal', 'Nizamabad', 'Hyderabad', 'Karimnagar', 'Hyderabad'],
        'Type': ['Government', 'private', 'Private', 'Government', 'govt', 'Government'],  # Inconsistent case
        'Bed_Capacity': [250, -50, None, 400, 150, 250],  # Negative and missing values
        'Established_Year': ['1995', '2010', '1980', '2005', '1990', '1995'],
        'Emergency_Services': ['Yes', 'yes', 'No', 'YES', '', 'Yes']  # Inconsistent values
    }
    
    df = pd.DataFrame(data)
    return df

def test_agents():
    """Test all agents with sample data"""
    
    print("🧪 Testing RTGS AI Analyst Agents...")
    
    # Create test data
    test_df = create_test_data()
    
    try:
        # Test Ingestion Agent
        print("\n1. Testing Data Ingestion Agent...")
        ingestion_agent = DataIngestionAgent()
        
        # Save test data to file and load it
        test_file = "test_data.csv"
        test_df.to_csv(test_file, index=False)
        loaded_df, metadata = ingestion_agent.load_csv(test_file)
        print(f"   ✅ Loaded {metadata['rows']} rows, {metadata['columns']} columns")
        
        # Test Inspector Agent
        print("\n2. Testing Inspector Agent...")
        inspector_agent = InspectorAgent()
        action_plan = inspector_agent.inspect_dataset(loaded_df)
        print(f"   ✅ Found {len(action_plan.get('null_values', {}))} columns with nulls")
        print(f"   ✅ Duplicates: {action_plan.get('duplicates', 'None')}")
        
        # Test Cleaning Agent
        print("\n3. Testing Cleaning Agent...")
        cleaning_agent = CleaningAgent()
        cleaned_df, cleaning_log = cleaning_agent.clean_dataset(loaded_df, action_plan)
        print(f"   ✅ Applied {len(cleaning_log)} cleaning operations")
        
        # Test Transforming Agent
        print("\n4. Testing Transforming Agent...")
        transforming_agent = TransformingAgent()
        transformed_df, transformation_log = transforming_agent.transform_dataset(cleaned_df)
        print(f"   ✅ Applied {len(transformation_log)} transformations")
        print(f"   ✅ Final shape: {transformed_df.shape}")
        
        # Test Verification Agent
        print("\n5. Testing Verification Agent...")
        verification_agent = VerificationAgent()
        success, verification_results = verification_agent.verify_dataset(
            loaded_df, transformed_df, cleaning_log, transformation_log
        )
        quality_score = verification_results.get('summary', {}).get('quality_score', 0)
        print(f"   ✅ Verification: {'PASSED' if success else 'FAILED'}")
        print(f"   ✅ Quality Score: {quality_score}/100")
        
        # Test Analysis Agent
        print("\n6. Testing Analysis Agent...")
        analysis_agent = AnalysisAgent("test_outputs")
        analysis_results = analysis_agent.generate_analysis(
            transformed_df, cleaning_log, transformation_log, verification_results
        )
        print(f"   ✅ Generated {len(analysis_results['policy_recommendations'])} policy recommendations")
        print(f"   ✅ Created {len(analysis_results['visualizations'])} visualizations")
        
        print("\n🎉 All agents tested successfully!")
        
        # Cleanup
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agents()
    sys.exit(0 if success else 1)