import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame from the provided data
data = {
    'latitude': [22.9781, 22.9732, 22.9755, 22.9718, 22.9823, 22.9689, 22.9767, 22.9740, 22.9801, 22.9695,
                 22.9773, 22.9726, 22.9830, 22.9678, 22.9759, 22.9707, 22.9812, 22.9748, 22.9762, 22.9711,
                 22.9793, 22.9683, 22.9778, 22.9737, 22.9820, 22.9699, 22.9769, 22.9720, 22.9807, 22.9672,
                 22.9752, 22.9703, 22.9816, 22.9744, 22.9765, 22.9715, 22.9797, 22.9687, 22.9785, 22.9730,
                 22.9827, 22.9691, 22.9757, 22.9709, 22.9809, 22.9675, 22.9749, 22.9713, 22.9818],
    'longitude': [88.4321, 88.4367, 88.4312, 88.4390, 88.4287, 88.4412, 88.4333, 88.4355, 88.4302, 88.4401,
                  88.4328, 88.4379, 88.4275, 88.4425, 88.4341, 88.4388, 88.4295, 88.4360, 88.4337, 88.4395,
                  88.4309, 88.4418, 88.4325, 88.4363, 88.4280, 88.4407, 88.4330, 88.4383, 88.4298, 88.4431,
                  88.4352, 88.4392, 88.4289, 88.4368, 88.4339, 88.4399, 88.4305, 88.4415, 88.4318, 88.4375,
                  88.4278, 88.4409, 88.4346, 88.4385, 88.4292, 88.4428, 88.4358, 88.4397, 88.4284],
    'safety_level': [0.82, 0.78, 0.85, 0.72, 0.88, 0.68, 0.83, 0.79, 0.87, 0.71,
                     0.84, 0.76, 0.89, 0.65, 0.81, 0.74, 0.86, 0.80, 0.82, 0.73,
                     0.85, 0.69, 0.83, 0.77, 0.88, 0.70, 0.84, 0.75, 0.86, 0.63,
                     0.81, 0.72, 0.87, 0.79, 0.82, 0.74, 0.85, 0.68, 0.84, 0.76,
                     0.89, 0.66, 0.80, 0.73, 0.86, 0.64, 0.81, 0.74, 0.87],
    'location_name': ['Kalyani University Campus', 'Kalyani Station Area', 'Kalyani Govt Quarters', 'Kalyani Bazaar',
                      'Kalyani Science Park', 'Kalyani Outer Road', 'Kalyani Central Park', 'Kalyani Main Road',
                      'Kalyani Research Area', 'Kalyani Industrial Belt', 'Kalyani Academic Zone',
                      'Kalyani Market Extension', 'Kalyani Green Valley', 'Kalyani Transport Nagar', 'Kalyani Township',
                      'Kalyani Old Quarter', 'Kalyani Tech Park', 'Kalyani Civic Center', 'Kalyani Library Area',
                      'Kalyani Wholesale Market', 'Kalyani Residential East', 'Kalyani Factory Zone',
                      'Kalyani Sports Complex', 'Kalyani Commercial Strip', 'Kalyani Botanical Garden',
                      'Kalyani Warehouse Area', 'Kalyani Administrative Zone', 'Kalyani Retail Hub',
                      'Kalyani Research Housing', 'Kalyani Industrial Extension', 'Kalyani Residential West',
                      'Kalyani Urban Core', 'Kalyani Innovation Center', 'Kalyani Service Road',
                      'Kalyani Cultural Center', 'Kalyani Trade Center', 'Kalyani Professors Colony',
                      'Kalyani Logistics Park', 'Kalyani University Housing', 'Kalyani Shopping District',
                      'Kalyani Eco Park', 'Kalyani Manufacturing Zone', 'Kalyani Township Extension',
                      'Kalyani Old Market', 'Kalyani Tech Housing', 'Kalyani Industrial North',
                      'Kalyani Civic Quarters', 'Kalyani Commercial North', 'Kalyani Science Housing'],
    'area_type': ['Educational', 'Commercial', 'Residential', 'Commercial', 'Institutional', 'Industrial',
                  'Recreational', 'Commercial', 'Institutional', 'Industrial', 'Educational', 'Commercial',
                  'Residential', 'Industrial', 'Residential', 'Residential', 'Institutional', 'Commercial',
                  'Educational', 'Commercial', 'Residential', 'Industrial', 'Recreational', 'Commercial',
                  'Recreational', 'Industrial', 'Institutional', 'Commercial', 'Residential', 'Industrial',
                  'Residential', 'Residential', 'Institutional', 'Commercial', 'Educational', 'Commercial',
                  'Residential', 'Industrial', 'Residential', 'Commercial', 'Recreational', 'Industrial',
                  'Residential', 'Commercial', 'Residential', 'Industrial', 'Residential', 'Commercial', 'Residential'],
    'population_density': ['Medium', 'High', 'Medium', 'Very High', 'Low', 'Medium', 'Low', 'High', 'Low', 'Medium',
                           'Medium', 'High', 'Low', 'High', 'Medium', 'High', 'Low', 'High', 'Medium', 'Very High',
                           'Medium', 'High', 'Low', 'High', 'Low', 'Medium', 'Medium', 'High', 'Low', 'High',
                           'Medium', 'Very High', 'Low', 'High', 'Medium', 'Very High', 'Medium', 'High', 'Medium',
                           'High', 'Low', 'High', 'Medium', 'High', 'Low', 'High', 'Medium', 'Very High', 'Low'],
    'crime_rate': [0.05, 0.18, 0.03, 0.25, 0.01, 0.30, 0.02, 0.15, 0.04, 0.28,
                   0.06, 0.20, 0.01, 0.35, 0.08, 0.22, 0.03, 0.12, 0.07, 0.27,
                   0.04, 0.32, 0.03, 0.17, 0.01, 0.29, 0.05, 0.19, 0.02, 0.38,
                   0.09, 0.24, 0.02, 0.14, 0.06, 0.26, 0.03, 0.33, 0.04, 0.16,
                   0.01, 0.36, 0.10, 0.21, 0.02, 0.37, 0.08, 0.23, 0.02],
    'street_lighting': ['Excellent', 'Good', 'Good', 'Moderate', 'Excellent', 'Poor', 'Excellent', 'Good', 'Excellent',
                        'Poor', 'Excellent', 'Moderate', 'Excellent', 'Poor', 'Good', 'Moderate', 'Excellent', 'Good',
                        'Excellent', 'Moderate', 'Good', 'Poor', 'Excellent', 'Good', 'Excellent', 'Poor', 'Excellent',
                        'Moderate', 'Excellent', 'Poor', 'Good', 'Moderate', 'Excellent', 'Good', 'Excellent',
                        'Moderate', 'Good', 'Poor', 'Excellent', 'Good', 'Excellent', 'Poor', 'Good', 'Moderate',
                        'Excellent', 'Poor', 'Good', 'Moderate', 'Excellent'],
    'police_presence': ['High', 'Moderate', 'High', 'Moderate', 'High', 'Low', 'Moderate', 'Moderate', 'High', 'Low',
                        'High', 'Moderate', 'High', 'Low', 'High', 'Moderate', 'High', 'High', 'High', 'Moderate',
                        'High', 'Low', 'Moderate', 'Moderate', 'High', 'Low', 'High', 'Moderate', 'High', 'Low',
                        'High', 'Moderate', 'High', 'Moderate', 'High', 'Moderate', 'High', 'Low', 'High', 'Moderate',
                        'High', 'Low', 'High', 'Moderate', 'High', 'Low', 'High', 'Moderate', 'High'],
    'emergency_response_time': [5, 8, 6, 10, 4, 15, 7, 9, 5, 14,
                                6, 11, 3, 16, 7, 12, 5, 8, 6, 13,
                                6, 15, 7, 10, 4, 14, 6, 11, 5, 17,
                                7, 12, 4, 9, 6, 13, 6, 15, 6, 10,
                                3, 16, 7, 12, 5, 17, 7, 12, 4],
    'public_transport': ['Good', 'Excellent', 'Moderate', 'Good', 'Poor', 'Moderate', 'Poor', 'Excellent', 'Moderate',
                         'Poor', 'Good', 'Good', 'Poor', 'Poor', 'Moderate', 'Good', 'Moderate', 'Excellent', 'Good',
                         'Good', 'Moderate', 'Poor', 'Poor', 'Excellent', 'Poor', 'Poor', 'Moderate', 'Good', 'Moderate',
                         'Poor', 'Moderate', 'Good', 'Moderate', 'Excellent', 'Good', 'Good', 'Moderate', 'Poor', 'Good',
                         'Excellent', 'Poor', 'Poor', 'Moderate', 'Good', 'Moderate', 'Poor', 'Moderate', 'Good', 'Moderate']
}

df = pd.DataFrame(data)

# Data Analysis Functions
def basic_analysis():
    print("=== Dataset Overview ===")
    print(f"Total locations: {len(df)}")
    print("\n=== Safety Level Statistics ===")
    print(df['safety_level'].describe())
    
    print("\n=== Area Type Distribution ===")
    print(df['area_type'].value_counts())
    
    print("\n=== Correlation Matrix ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].corr())

def visualize_data():
    plt.figure(figsize=(15, 10))
    
    # Safety Level Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['safety_level'], bins=10, kde=True)
    plt.title('Safety Level Distribution')
    
    # Safety by Area Type
    plt.subplot(2, 2, 2)
    sns.boxplot(x='area_type', y='safety_level', data=df)
    plt.title('Safety Level by Area Type')
    plt.xticks(rotation=45)
    
    # Crime Rate vs Safety
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='crime_rate', y='safety_level', hue='police_presence', data=df)
    plt.title('Crime Rate vs Safety Level')
    
    # Emergency Response Time vs Safety
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='emergency_response_time', y='safety_level', hue='street_lighting', data=df)
    plt.title('Response Time vs Safety Level')
    
    plt.tight_layout()
    plt.show()

def map_visualization():
    plt.figure(figsize=(12, 8))
    plt.scatter(df['longitude'], df['latitude'], c=df['safety_level'], cmap='RdYlGn', s=100)
    plt.colorbar(label='Safety Level')
    
    # Annotate some important locations
    for i, row in df[df['safety_level'] > 0.85].iterrows():
        plt.annotate(row['location_name'], (row['longitude'], row['latitude']), 
                     textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    
    plt.title('Kalyani Safety Level Heatmap')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

def find_safest_areas():
    print("\n=== Top 5 Safest Areas ===")
    print(df.nlargest(5, 'safety_level')[['location_name', 'safety_level', 'area_type']])
    
    print("\n=== Top 5 Least Safe Areas ===")
    print(df.nsmallest(5, 'safety_level')[['location_name', 'safety_level', 'area_type']])

def analyze_factors():
    print("\n=== Safety Level by Police Presence ===")
    print(df.groupby('police_presence')['safety_level'].mean())
    
    print("\n=== Safety Level by Street Lighting ===")
    print(df.groupby('street_lighting')['safety_level'].mean())
    
    print("\n=== Safety Level by Public Transport ===")
    print(df.groupby('public_transport')['safety_level'].mean())

# Execute analysis
basic_analysis()
visualize_data()
map_visualization()
find_safest_areas()
analyze_factors()

# Additional Feature: Create safety score based on multiple factors
def calculate_safety_score(row):
    # Weights for different factors (can be adjusted)
    weights = {
        'crime_rate': -0.4,
        'police_presence': {'Low': 0.1, 'Moderate': 0.3, 'High': 0.5},
        'street_lighting': {'Poor': 0.1, 'Moderate': 0.3, 'Good': 0.4, 'Excellent': 0.5},
        'emergency_response_time': -0.2,
        'public_transport': {'Poor': 0.1, 'Moderate': 0.2, 'Good': 0.3, 'Excellent': 0.4}
    }
    
    score = 0
    score += row['crime_rate'] * weights['crime_rate']
    score += weights['police_presence'][row['police_presence']]
    score += weights['street_lighting'][row['street_lighting']]
    score += (row['emergency_response_time'] / 20) * weights['emergency_response_time']
    score += weights['public_transport'][row['public_transport']]
    
    # Normalize to 0-1 scale
    return (score - 0.2) / 1.2  # Adjusted based on min-max possible values

df['calculated_safety_score'] = df.apply(calculate_safety_score, axis=1)

# Compare calculated score with provided safety level
print("\n=== Calculated Safety Score vs Provided Safety Level ===")
print(df[['location_name', 'safety_level', 'calculated_safety_score']].head(10))
ml_dataset.to_csv('kalyani_safety_dataset.csv', index=False)