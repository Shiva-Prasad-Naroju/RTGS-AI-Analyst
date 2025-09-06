# RTGS AI Analyst - Healthcare Data Analysis Report

## Dataset Summary

- **Shape**: 31 rows × 37 columns
- **Data Quality Score**: 100/100 (Grade: A)
- **Processing Operations**: 17 total
  - Cleaning: 12
  - Transformations: 5

## Key Healthcare Metrics

- **Total Beds**: 20528
- **Average Beds Per Hospital**: 662.19
- **Hospitals With Zero Beds**: 0

## Geographic Distribution

**Total Locations**: 2

### Top Locations by Hospital Count
- 0: 25 hospitals
- 1: 6 hospitals

### Potentially Underserved Locations
- 0: 25 hospitals
- 1: 6 hospitals

## Policy Recommendations

1. Address geographic inequality: 1 locations have significantly fewer hospitals. Priority areas: 1
2. Standardize hospital capacity: High variation in bed capacity suggests need for capacity planning

## Technical Analysis Details

### Strong Correlations Found
- District_Hospitals <-> Health_Sub_Centres_Mean_by_District_Hospitals: 1.0
- District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals: 1.0
- District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals: 1.0
- District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals: 1.0
- District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals: 1.0
- District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals: -1.0
- District_Hospitals <-> District_Hospitals_Normalized: 1.0
- District_Hospitals <-> Health_Sub_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: 1.0
- District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals: -1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Health_Sub_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals: -1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Health_Sub_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals: -1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Health_Sub_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals: 1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals: -1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> Health_Sub_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals: -1.0
- Community_Health_Centres_Mean_by_District_Hospitals <-> District_Hospitals_Normalized: 1.0
- Community_Health_Centres_Mean_by_District_Hospitals <-> Health_Sub_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Community_Health_Centres_Mean_by_District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Community_Health_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Community_Health_Centres_Mean_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Community_Health_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Community_Health_Centres_Mean_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Std_by_District_Hospitals <-> District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Std_by_District_Hospitals <-> Health_Sub_Centres_Mean_by_District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Std_by_District_Hospitals <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Std_by_District_Hospitals <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Std_by_District_Hospitals <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Std_by_District_Hospitals <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- District_Hospitals_Normalized <-> Health_Sub_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- District_Hospitals_Normalized <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: 1.0
- District_Hospitals_Normalized <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- District_Hospitals_Normalized <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- District_Hospitals_Normalized <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- District_Hospitals_Normalized <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Health_Sub_Centres_Mean_by_District_Hospitals_Normalized <-> Health_Sub_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals_Normalized <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals_Normalized <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals_Normalized <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Mean_by_District_Hospitals_Normalized <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Health_Sub_Centres_Std_by_District_Hospitals_Normalized <-> Primary_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals_Normalized <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals_Normalized <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Health_Sub_Centres_Std_by_District_Hospitals_Normalized <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Primary_Health_Centres_Mean_by_District_Hospitals_Normalized <-> Primary_Health_Centres_Std_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals_Normalized <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Mean_by_District_Hospitals_Normalized <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Primary_Health_Centres_Std_by_District_Hospitals_Normalized <-> Community_Health_Centres_Mean_by_District_Hospitals_Normalized: 1.0
- Primary_Health_Centres_Std_by_District_Hospitals_Normalized <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Community_Health_Centres_Mean_by_District_Hospitals_Normalized <-> Community_Health_Centres_Std_by_District_Hospitals_Normalized: -1.0
- Beds_in_all_Hospitals <-> Beds_in_all_Hospitals_Normalized: 1.0
- Beds_in_all_Hospitals <-> Primary_Health_Centres_Normalized: 0.809
- Beds_in_all_Hospitals <-> Teaching_Hospitals_Normalized: 0.899
- Beds_in_all_Hospitals <-> Doctors_in_all_Hospitals_Normalized: 0.91
- Beds_in_all_Hospitals <-> Primary_Health_Centres: 0.809
- Beds_in_all_Hospitals <-> Teaching_Hospitals: 0.899
- Beds_in_all_Hospitals <-> Doctors_in_all_Hospitals: 0.91
- Beds_in_all_Hospitals_Normalized <-> Primary_Health_Centres_Normalized: 0.809
- Beds_in_all_Hospitals_Normalized <-> Teaching_Hospitals_Normalized: 0.899
- Beds_in_all_Hospitals_Normalized <-> Doctors_in_all_Hospitals_Normalized: 0.91
- Beds_in_all_Hospitals_Normalized <-> Primary_Health_Centres: 0.809
- Beds_in_all_Hospitals_Normalized <-> Teaching_Hospitals: 0.899
- Beds_in_all_Hospitals_Normalized <-> Doctors_in_all_Hospitals: 0.91
- Health_Sub_Centres_Normalized <-> Health_Sub_Centres: 1.0
- Primary_Health_Centres_Normalized <-> Teaching_Hospitals_Normalized: 0.769
- Primary_Health_Centres_Normalized <-> Doctors_in_all_Hospitals_Normalized: 0.892
- Primary_Health_Centres_Normalized <-> Primary_Health_Centres: 1.0
- Primary_Health_Centres_Normalized <-> Teaching_Hospitals: 0.769
- Primary_Health_Centres_Normalized <-> Doctors_in_all_Hospitals: 0.892
- Community_Health_Centres_Normalized <-> Community_Health_Centres: 1.0
- Area_Hospitals_Normalized <-> Area_Hospitals: 1.0
- Teaching_Hospitals_Normalized <-> Doctors_in_all_Hospitals_Normalized: 0.908
- Teaching_Hospitals_Normalized <-> Primary_Health_Centres: 0.769
- Teaching_Hospitals_Normalized <-> Teaching_Hospitals: 1.0
- Teaching_Hospitals_Normalized <-> Doctors_in_all_Hospitals: 0.908
- Ayurveda_Hospitals_incl_Dispensaries_Normalized <-> Ayurveda_Hospitals_incl_Dispensaries: 1.0
- Homeopathic_Hospitals_incl_Dispensaries_Normalized <-> Homeopathic_Hospitals_incl_Dispensaries: 1.0
- Unani_Hospitals_incl_Dispensaries_Normalized <-> Unani_Hospitals_incl_Dispensaries: 1.0
- Naturopathy_Hospitals_incl_Dispensaries_Normalized <-> Naturopathy_Hospitals_incl_Dispensaries: 1.0
- Doctors_in_all_Hospitals_Normalized <-> Primary_Health_Centres: 0.892
- Doctors_in_all_Hospitals_Normalized <-> Teaching_Hospitals: 0.908
- Doctors_in_all_Hospitals_Normalized <-> Doctors_in_all_Hospitals: 1.0
- Primary_Health_Centres <-> Teaching_Hospitals: 0.769
- Primary_Health_Centres <-> Doctors_in_all_Hospitals: 0.892
- Teaching_Hospitals <-> Doctors_in_all_Hospitals: 0.908

## Processing Summary

- **Total cleaning operations**: 12
- **Total transformations**: 5
- **Final quality score**: 100/100
- **Warnings**: 0
- **Errors**: 0

