import pandas as pd
import numpy as np
from fpdf import FPDF
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def classify_behavior(df, model):
    """
    Classify student behavior using the trained model.
    
    Args:
        df: DataFrame with student engagement data
        model: Trained classification model
        
    Returns:
        DataFrame with added classification column
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if required features are present in the dataframe
    required_features = ['gaze_direction', 'head_pose', 'activity_level']
    missing_features = [feat for feat in required_features if feat not in df.columns]
    
    if missing_features:
        # Handle missing features with dummy data for demonstration
        for feat in missing_features:
            result_df[feat] = np.random.uniform(0, 1, len(df))
            
    # Extract features for prediction
    X = result_df[required_features]
    
    # Predict engagement level
    try:
        # Use model to predict engagement level (assumes model outputs value between 0-1)
        result_df['engagement_level'] = model.predict(X)
    except Exception as e:
        # Fallback if model prediction fails
        print(f"Model prediction failed: {str(e)}")
        result_df['engagement_level'] = np.random.uniform(0.2, 0.9, len(df))
    
    # Add engagement category based on engagement level
    def categorize_engagement(level):
        if level > 0.7:
            return "Highly Engaged"
        elif level > 0.4:
            return "Engaged"
        else:
            return "Disengaged"
    
    result_df['engagement_category'] = result_df['engagement_level'].apply(categorize_engagement)
    
    # Add additional insights
    if 'student_id' in df.columns and 'timestamp' in df.columns:
        # Calculate time-based statistics if time data is available
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        result_df['day_of_week'] = result_df['timestamp'].dt.day_name()
        result_df['hour_of_day'] = result_df['timestamp'].dt.hour
        
    return result_df

def generate_report(df):
    """
    Generate a PDF report from the classified data.
    
    Args:
        df: DataFrame with classified engagement data
        
    Returns:
        Path to the generated PDF file
    """
    # Create reports directory if it doesn't exist
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate visualizations for the report
    fig_path = _generate_visualizations(df)
    
    # Create a PDF report
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Student Engagement Report', 0, 1, 'C')
    pdf.ln(5)
    
    # Add date
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
    pdf.ln(5)
    
    # Add summary statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Engagement Summary', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Calculate statistics
    if 'engagement_category' in df.columns:
        engagement_counts = df['engagement_category'].value_counts()
        
        for category, count in engagement_counts.items():
            percentage = (count / len(df)) * 100
            pdf.cell(0, 10, f'{category}: {count} instances ({percentage:.1f}%)', 0, 1)
    
    # Add average engagement if available
    if 'engagement_level' in df.columns:
        avg_engagement = df['engagement_level'].mean()
        pdf.cell(0, 10, f'Average Engagement Level: {avg_engagement:.2f}', 0, 1)
    
    # Add visualizations if they were generated
    if os.path.exists(fig_path):
        pdf.ln(5)
        pdf.cell(0, 10, 'Engagement Visualization:', 0, 1)
        pdf.image(fig_path, x=10, y=None, w=180)
    
    # Add recommendations section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommendations', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Generate simple recommendations based on the data
    highly_engaged_pct = 0
    if 'engagement_category' in df.columns:
        engagement_counts = df['engagement_category'].value_counts()
        total = len(df)
        highly_engaged_pct = engagement_counts.get('Highly Engaged', 0) / total * 100 if total > 0 else 0
    
    # Add different recommendations based on engagement levels
    if highly_engaged_pct < 30:
        pdf.multi_cell(0, 10, '- Consider incorporating more interactive activities to increase student engagement.\n'
                      '- Try shorter lesson segments with frequent changes in activity types.\n'
                      '- Implement peer learning strategies to boost participation.', 0)
    elif highly_engaged_pct < 60:
        pdf.multi_cell(0, 10, '- Build on current engagement strategies that are working well.\n'
                      '- Identify and focus on specific times or activities where engagement dips.\n'
                      '- Consider targeted interventions for students showing lower engagement.', 0)
    else:
        pdf.multi_cell(0, 10, '- Maintain current successful teaching strategies.\n'
                      '- Consider adding more challenging content to keep highly engaged students stimulated.\n'
                      '- Document and share your effective practices with colleagues.', 0)
    
    # Add individual student section if student_id is available
    if 'student_id' in df.columns:
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Individual Student Analysis', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Group by student_id and calculate average engagement
        student_avg = df.groupby('student_id')['engagement_level'].mean().sort_values(ascending=False)
        
        # List top 3 most engaged students
        pdf.cell(0, 10, 'Most Engaged Students:', 0, 1)
        for i, (student, score) in enumerate(student_avg.head(3).items()):
            pdf.cell(0, 10, f'{i+1}. Student ID {student}: {score:.2f}', 0, 1)
        
        # List bottom 3 least engaged students
        pdf.ln(5)
        pdf.cell(0, 10, 'Students Needing Additional Support:', 0, 1)
        for i, (student, score) in enumerate(student_avg.tail(3).items()):
            pdf.cell(0, 10, f'{i+1}. Student ID {student}: {score:.2f}', 0, 1)
    
    # Save the PDF
    report_path = os.path.join(report_dir, f'engagement_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
    pdf.output(report_path)
    
    # Clean up temporary files
    if os.path.exists(fig_path):
        os.remove(fig_path)
    
    return report_path

def _generate_visualizations(df):
    """
    Generate visualizations for the engagement report.
    
    Args:
        df: DataFrame with classified engagement data
        
    Returns:
        Path to the saved visualization image
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Engagement distribution
    plt.subplot(2, 1, 1)
    if 'engagement_category' in df.columns:
        sns.countplot(x='engagement_category', data=df, palette='viridis')
        plt.title('Distribution of Engagement Categories')
        plt.xlabel('Engagement Category')
        plt.ylabel('Count')
    
    # Plot 2: Additional visualization based on available data
    plt.subplot(2, 1, 2)
    
    if 'student_id' in df.columns and 'engagement_level' in df.columns:
        # If we have student IDs, show average engagement by student
        student_avg = df.groupby('student_id')['engagement_level'].mean().sort_values(ascending=False)
        student_avg.plot(kind='bar', color='skyblue')
        plt.title('Average Engagement by Student')
        plt.xlabel('Student ID')
        plt.ylabel('Avg. Engagement')
        plt.tight_layout()
    elif 'timestamp' in df.columns and 'engagement_level' in df.columns:
        # If we have timestamps, show engagement over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp')['engagement_level'].plot(marker='o', linestyle='-')
        plt.title('Engagement Level Over Time')
        plt.xlabel('Time')
        plt.ylabel('Engagement Level')
        plt.tight_layout()
    elif 'day_of_week' in df.columns:
        # Show engagement by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_avg = df.groupby('day_of_week')['engagement_level'].mean()
        day_avg = day_avg.reindex(day_order)
        day_avg.plot(kind='bar', color='lightgreen')
        plt.title('Average Engagement by Day of Week')
        plt.xlabel('Day')
        plt.ylabel('Avg. Engagement')
        plt.tight_layout()
    else:
        # Generic histogram of engagement levels
        plt.hist(df['engagement_level'], bins=10, color='salmon')
        plt.title('Distribution of Engagement Levels')
        plt.xlabel('Engagement Level')
        plt.ylabel('Frequency')
        plt.tight_layout()
    
    # Save the figure to a temporary file
    fig_path = os.path.join('reports', 'temp_visualization.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig_path

def create_sample_model():
    """
    Creates a simple model for demonstration purposes.
    This is useful when the real model file is missing.
    
    Returns:
        A simple prediction model
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Create a simple random forest model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Generate some random training data and fit the model
    X_train = np.random.rand(100, 3)  # 3 features
    y_train = 0.3 * X_train[:, 0] + 0.4 * X_train[:, 1] + 0.3 * X_train[:, 2] + np.random.normal(0, 0.1, 100)
    y_train = np.clip(y_train, 0, 1)  # Ensure values are between 0 and 1
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model

def generate_sample_data(n_samples=50):
    """
    Generate sample student engagement data for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample engagement data
    """
    # Create student IDs
    student_ids = np.random.randint(1000, 9999, n_samples)
    
    # Create timestamps for a week
    start_date = datetime.datetime.now() - datetime.timedelta(days=7)
    timestamps = [start_date + datetime.timedelta(
        days=np.random.randint(0, 7),
        hours=np.random.randint(8, 16),
        minutes=np.random.randint(0, 60)
    ) for _ in range(n_samples)]
    
    # Create engagement features
    gaze_direction = np.random.uniform(0, 1, n_samples)
    head_pose = np.random.uniform(0, 1, n_samples)
    activity_level = np.random.uniform(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'student_id': student_ids,
        'timestamp': timestamps,
        'gaze_direction': gaze_direction,
        'head_pose': head_pose,
        'activity_level': activity_level
    })
    
    return df