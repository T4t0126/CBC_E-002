import streamlit as st
import os
import sys
from streamlit.components.v1 import html
import pandas as pd
import pickle
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
# Add this directory to path to make local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Try to import the utility functions
try:
    from utils import classify_behavior, generate_report, create_sample_model, generate_sample_data
    utils_imported = True
except ImportError:
    utils_imported = False
    st.error("Failed to import utility functions. Make sure utils.py exists in the same directory.")
# Set page configuration
st.set_page_config(
    page_title="AI Classroom Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Create sidebar for navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to", ["Live Monitoring", "Data Analysis", "Settings", "Help"])
# Function to find the model file
def load_model():
    """Attempt to load the machine learning model from various locations"""
    # List of possible locations to check
    possible_paths = [
        'engagement_model.pkl',
        'models/engagement_model.pkl',
        '../models/engagement_model.pkl',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engagement_model.pkl'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'engagement_model.pkl')
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as model_file:
                    return pickle.load(model_file)
            except Exception as e:
                st.warning(f"Found model at {path} but failed to load it: {str(e)}")

    # No model found, try to create a sample model
    if utils_imported:
        st.warning("No model found. Creating a sample model for demonstration purposes.")
        return create_sample_model()

    return None
# Try to load the model
model = load_model()
# Display model status in sidebar
with st.sidebar.expander("System Status"):
    if model is None:
        st.warning("⚠️ Model not found. Some features will be limited.")
        demo_mode = True
    else:
        st.success("✅ Model loaded successfully!")
        demo_mode = False

    if utils_imported:
        st.success("✅ Utility functions loaded!")
    else:
        st.error("❌ Utility functions not found!")
# Add demo mode toggle if in demo mode
if demo_mode and utils_imported:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Demo Options")
    use_demo = st.sidebar.checkbox("Use demonstration data", value=True)
# Add time information
st.sidebar.markdown("---")
now = datetime.datetime.now()
st.sidebar.write(f"Current date: {now.strftime('%Y-%m-%d')}")
st.sidebar.write(f"Current time: {now.strftime('%H:%M:%S')}")
# Live Monitoring Page
if page == "Live Monitoring":
    st.title("📚 AI Classroom Engagement Monitor")
    st.markdown("### 👀 Live Engagement Detection")

    # Display the webcam component if the HTML file exists
    webcam_path = "ai_class\frontend\components\webcam.html"
    if os.path.exists(webcam_path):
        try:
            with open(webcam_path, "r") as f:
                html(f.read(), height=400)
        except Exception as e:
            st.error(f"Error loading webcam component: {str(e)}")
    else:
        st.error(f"Webcam component not found at {webcam_path}")
        if st.button("Create Sample Webcam Component"):
            try:
                os.makedirs(os.path.dirname(webcam_path), exist_ok=True)
                with open(webcam_path, "w") as f:
                    f.write("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Webcam Feed</title>
                        <style>
                            body {
                                margin: 0;
                                padding: 0;
                                font-family: Arial, sans-serif;
                            }
                            .container {
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                            }
                            .webcam-container {
                                position: relative;
                                width: 640px;
                                height: 480px;
                                margin: 10px;
                                border: 2px solid #4CAF50;
                                border-radius: 8px;
                                overflow: hidden;
                            }
                            .video {
                                width: 100%;
                                height: 100%;
                                object-fit: cover;
                            }
                            .controls {
                                display: flex;
                                justify-content: center;
                                margin-top: 10px;
                            }
                            button {
                                margin: 0 5px;
                                padding: 8px 15px;
                                border: none;
                                border-radius: 4px;
                                background-color: #4CAF50;
                                color: white;
                                cursor: pointer;
                                font-size: 14px;
                            }
                            button:hover {
                                background-color: #45a049;
                            }
                            .detection-box {
                                position: absolute;
                                border: 3px solid red;
                                display: none;
                            }
                            .status {
                                margin-top: 10px;
                                padding: 8px;
                                border-radius: 4px;
                                text-align: center;
                                font-weight: bold;
                            }
                            .engaged {
                                background-color: rgba(76, 175, 80, 0.3);
                                color: #2e7d32;
                            }
                            .distracted {
                                background-color: rgba(244, 67, 54, 0.3);
                                color: #c62828;
                            }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="webcam-container">
                                <video id="webcam" class="video" autoplay playsinline></video>
                                <div id="face-box" class="detection-box"></div>
                            </div>
                            <div class="controls">
                                <button id="start-btn">Start Camera</button>
                                <button id="stop-btn">Stop Camera</button>
                            </div>
                            <div id="status" class="status engaged">Student is engaged</div>
                        </div>

                        <script>
                            const video = document.getElementById('webcam');
                            const startButton = document.getElementById('start-btn');
                            const stopButton = document.getElementById('stop-btn');
                            const statusDiv = document.getElementById('status');
                            const faceBox = document.getElementById('face-box');

                            let stream = null;
                            let engagementStates = ['engaged', 'distracted'];
                            let detectionInterval;

                            // Start webcam
                            startButton.addEventListener('click', async () => {
                                try {
                                    stream = await navigator.mediaDevices.getUserMedia({
                                        video: true,
                                        audio: false
                                    });
                                    video.srcObject = stream;

                                    // Start simulated engagement detection
                                    startDetection();
                                } catch (err) {
                                    console.error("Error accessing webcam:", err);
                                    statusDiv.textContent = "Error accessing webcam. Please check permissions.";
                                    statusDiv.className = "status distracted";
                                }
                            });

                            // Stop webcam
                            stopButton.addEventListener('click', () => {
                                if (stream) {
                                    stream.getTracks().forEach(track => track.stop());
                                    video.srcObject = null;
                                    stopDetection();
                                }
                            });

                            // Simulate engagement detection
                            function startDetection() {
                                detectionInterval = setInterval(() => {
                                    // Simulate random engagement/distraction states
                                    const isEngaged = Math.random() > 0.3;

                                    // Update UI
                                    statusDiv.textContent = isEngaged ? "Student is engaged" : "Student is distracted";
                                    statusDiv.className = isEngaged ? "status engaged" : "status distracted";

                                    // Show simulated face detection box
                                    const videoWidth = video.clientWidth;
                                    const videoHeight = video.clientHeight;

                                    // Random position and size for face box
                                    const boxWidth = Math.round(videoWidth * 0.15 + Math.random() * 30);
                                    const boxHeight = Math.round(boxWidth * 1.3);
                                    const boxLeft = Math.round(Math.random() * (videoWidth - boxWidth));
                                    const boxTop = Math.round(Math.random() * (videoHeight - boxHeight));

                                    faceBox.style.width = boxWidth + 'px';
                                    faceBox.style.height = boxHeight + 'px';
                                    faceBox.style.left = boxLeft + 'px';
                                    faceBox.style.top = boxTop + 'px';
                                    faceBox.style.display = 'block';
                                    faceBox.style.borderColor = isEngaged ? '#4CAF50' : '#F44336';

                                    // Send the state to Streamlit
                                    if (window.parent && window.parent.postMessage) {
                                        window.parent.postMessage({
                                            type: 'engagement',
                                            engaged: isEngaged
                                        }, '*');
                                    }
                                }, 1500);
                            }

                            function stopDetection() {
                                clearInterval(detectionInterval);
                                faceBox.style.display = 'none';
                                statusDiv.textContent = "Detection stopped";
                                statusDiv.className = "status";
                            }
                        </script>
                    </body>
                    </html>
                    """)
                st.success(f"Created webcam component at {webcam_path}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create webcam component: {str(e)}")

    # Simulated live data
    st.markdown("### 📊 Real-time Student Engagement")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Overview")
        # Create placeholder for real-time metrics
        metrics_placeholder = st.empty()

        # In a real app, this would be updated with actual data
        if demo_mode and utils_imported:
            engagement_percentage = np.random.randint(70, 95)
            metrics_placeholder.metric(
                label="Current Engagement Level",
                value=f"{engagement_percentage}%",
                delta=f"{np.random.randint(-5, 6)}%"
            )

            # Display student statistics
            st.write("**Student Statistics:**")
            col_stats1, col_stats2 = st.columns(2)

            with col_stats1:
                st.metric("Students Present", f"{np.random.randint(15, 25)}/25")
                st.metric("Active Participation", f"{np.random.randint(60, 90)}%")

            with col_stats2:
                st.metric("Average Attention Span", f"{np.random.randint(7, 12)} min")
                st.metric("Questions Asked", f"{np.random.randint(3, 10)}")

    with col2:
        st.subheader("Live Alerts")
        alert_placeholder = st.empty()

        # Simulated alerts
        if demo_mode and utils_imported:
            alert_types = [
                "⚠️ Low engagement detected in back row",
                "📱 Possible phone usage detected (Student ID: S12)",
                "😴 Drowsiness detected (Student ID: S07)",
                "👍 Engagement levels recovering after group activity"
            ]

            # Create a container for alerts
            with alert_placeholder.container():
                for i in range(min(4, np.random.randint(1, 5))):
                    alert = np.random.choice(alert_types)
                    alert_time = (datetime.datetime.now() - datetime.timedelta(minutes=np.random.randint(0, 10))).strftime("%H:%M:%S")
                    st.warning(f"{alert_time} - {alert}")

    # Create an engagement timeline chart
    st.markdown("### 📈 Engagement Timeline (Last 30 Minutes)")

    chart_placeholder = st.empty()

    if demo_mode and utils_imported:
        # Generate sample engagement data
        timestamps = pd.date_range(end=datetime.datetime.now(), periods=30, freq='1min')
        engagement_levels = np.clip(np.cumsum(np.random.normal(0, 5, 30)) + 75, 30, 100)

        # Create DataFrame
        engagement_df = pd.DataFrame({
            'Time': timestamps,
            'Engagement Level': engagement_levels
        })

        # Plot the chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(engagement_df['Time'], engagement_df['Engagement Level'], marker='o', linestyle='-', color='#4CAF50')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Engagement Level (%)')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)

        # Highlight current time
        ax.axvline(x=timestamps.max(), color='blue', linestyle=':', alpha=0.7)

        # Annotate regions
        ax.fill_between(engagement_df['Time'], 0, 30, alpha=0.2, color='red', label='Low Engagement')
        ax.fill_between(engagement_df['Time'], 30, 70, alpha=0.2, color='yellow', label='Moderate Engagement')
        ax.fill_between(engagement_df['Time'], 70, 100, alpha=0.2, color='green', label='High Engagement')

        ax.legend()
        chart_placeholder.pyplot(fig)

    # Add automated recommendations based on engagement levels
    st.markdown("### 🧠 AI Recommendations")

    if demo_mode and utils_imported:
        if np.random.random() > 0.5:
            st.info("💡 **Recommendation**: Engagement is declining. Consider introducing an interactive activity in the next 5-10 minutes.")
        else:
            st.info("💡 **Recommendation**: Students in the back-right section show lower engagement. Consider directing questions to that area.")

    # Add a manual intervention section
    with st.expander("📝 Record Manual Intervention"):
        intervention_type = st.selectbox(
            "Intervention Type",
            ["Group Activity", "Direct Question", "Break", "Content Change", "Gamification", "Other"]
        )

        intervention_notes = st.text_area("Notes", placeholder="Describe the intervention and its immediate effects...")

        if st.button("Record Intervention"):
            st.success("Intervention recorded successfully!")

# Data Analysis Page
elif page == "Data Analysis":
    st.title("📊 Data Analysis and Reports")

    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Historical Data", "Student Profiles", "Generate Reports"])

    with tab1:
        st.header("Historical Engagement Data")

        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.date.today() - datetime.timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.date.today()
            )

        # Class selector
        class_options = ["All Classes", "Math 101", "Physics 202", "Computer Science 110", "English Literature 240"]
        selected_class = st.selectbox("Select Class", class_options)

        # Generate sample data for demonstration
        if demo_mode and utils_imported:
            # Generate dates between start and end date
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Generate engagement data
            engagement_data = []
            for date in date_range:
                if date.weekday() < 5:  # Weekdays only
                    base_engagement = 75 + np.random.normal(0, 5)

                    # Add different patterns based on day of week
                    if date.weekday() == 0:  # Monday
                        base_engagement -= 5
                    if date.weekday() == 4:  # Friday
                        base_engagement -= 8

                    engagement_data.append({
                        'Date': date,
                        'Class': np.random.choice(class_options[1:]),
                        'Average Engagement': min(max(base_engagement, 40), 95),
                        'Peak Engagement': min(base_engagement + np.random.uniform(5, 15), 100),
                        'Low Engagement': max(base_engagement - np.random.uniform(15, 30), 20),
                        'Participation Rate': min(max(base_engagement - 10 + np.random.normal(0, 8), 30), 95),
                    })

            # Create dataframe
            df = pd.DataFrame(engagement_data)

            # Filter by selected class
            if selected_class != "All Classes":
                df = df[df['Class'] == selected_class]

            # Display the data
            st.write(f"Showing data for {len(df)} class sessions")
            st.dataframe(df)

            # Create visualization
            st.subheader("Engagement Trends")

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(df['Date'], df['Average Engagement'], marker='o', label='Average', linewidth=2)
            ax.plot(df['Date'], df['Peak Engagement'], marker='^', linestyle='--', alpha=0.7, label='Peak')
            ax.plot(df['Date'], df['Low Engagement'], marker='v', linestyle='--', alpha=0.7, label='Lowest')

            ax.set_xlabel('Date')
            ax.set_ylabel('Engagement Level (%)')
