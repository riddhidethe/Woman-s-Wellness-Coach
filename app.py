import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Mock platform class for demonstration - simplified version
class MultiModalHealthPlatform:
    def __init__(self):
        self.config = {'num_clusters': 5}
        
    def preprocess_data(self, health_data, notes, wearable_data, preferences):
        # Simplified preprocessing that combines all previous methods
        return np.random.random((1, 20))
            
    def generate_clusters(self, integrated_data):
        # Combined clustering method
        cluster_id = np.random.randint(0, 5)
        return cluster_id
            
    def generate_health_recommendations(self, patient_id, cluster_id):
        # Get recommendations based on cluster
        cluster_str = str(cluster_id)
        
        recommendations = {
            'patient_id': patient_id,
            'cluster': int(cluster_id),
            'primary': RECOMMENDATION_RULES[cluster_str]['primary'],
            'screenings': RECOMMENDATION_RULES[cluster_str]['screenings'],
            'lifestyle': RECOMMENDATION_RULES[cluster_str]['lifestyle']
        }
        return recommendations
            
    def visualize_health_profile(self, cluster_id):

        # Define health dimensions with corresponding icons and meaningful descriptions
        health_dimensions = [
            {"name": "Physical Health", "icon": "💪", "description": "Overall physical fitness and medical indicators"},
            {"name": "Mental Wellness", "icon": "🧘", "description": "Emotional balance and stress management"},
            {"name": "Preventive Care", "icon": "🩺", "description": "Health screenings and proactive medical checks"},
            {"name": "Nutrition", "icon": "🥗", "description": "Diet quality and nutritional balance"},
            {"name": "Physical Activity", "icon": "🏃‍♀️", "description": "Exercise and movement patterns"}
        ]

        # Generate random scores (in a real scenario, these would be data-driven)
        np.random.seed(cluster_id)  # Ensure consistent results for same cluster
        scores = np.random.uniform(0.5, 0.9, 5)

        # Create a figure that mimics a health report card
        fig, ax = plt.subplots(figsize=(10, 7), facecolor='#fcf7fa')
        plt.title('Your Wellness Profile', fontsize=16, color='#e75480', fontweight='bold', pad=20)

        # Color palette
        base_color = '#e75480'
        background_color = '#fcf7fa'
        text_color = '#333333'

        # Remove axes
        ax.set_axis_off()

        # Create a "card" look
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=base_color, linewidth=2, transform=ax.transAxes))

        for i, (dimension, score) in enumerate(zip(health_dimensions, scores)):
            # Vertical position
            y_position = 0.8 - (i * 0.15)

            # Icon and text
            plt.text(0.05, y_position, dimension['icon'], fontsize=20, transform=ax.transAxes, 
                    verticalalignment='center', color=base_color)
            plt.text(0.15, y_position, dimension['name'], fontsize=12, transform=ax.transAxes, 
                    verticalalignment='center', color=text_color, fontweight='bold')
            plt.text(0.15, y_position-0.03, dimension['description'], fontsize=8, transform=ax.transAxes, 
                    verticalalignment='center', color='#666666')

            # Progress bar background
            ax.add_patch(plt.Rectangle((0.5, y_position), 0.4, 0.05, 
                        facecolor='#e0e0e0', edgecolor='none', transform=ax.transAxes))
            
            # Progress bar foreground
            ax.add_patch(plt.Rectangle((0.5, y_position), 0.4 * score, 0.05, 
                        facecolor=base_color, edgecolor='none', alpha=0.7, transform=ax.transAxes))
            
            # Percentage text
            plt.text(0.92, y_position, f"{int(score*100)}%", fontsize=10, transform=ax.transAxes, 
                    verticalalignment='center', horizontalalignment='right', color=text_color)

        # Overall wellness score
        overall_score = np.mean(scores)
        plt.text(0.5, 0.05, f"Overall Wellness Score: {int(overall_score*100)}%", 
                transform=ax.transAxes, horizontalalignment='center', 
                fontsize=12, fontweight='bold', color=base_color)

        # Wellness level interpretation
        wellness_levels = {
            (0, 0.3): "Needs Attention",
            (0.3, 0.6): "Developing",
            (0.6, 0.8): "Good",
            (0.8, 1.0): "Excellent"
        }
        
        for (low, high), level in wellness_levels.items():
            if low <= overall_score < high:
                plt.text(0.5, 0.02, f"Wellness Level: {level}", 
                        transform=ax.transAxes, horizontalalignment='center', 
                        fontsize=10, color='#666666')
                break

        plt.tight_layout(pad=3)
        return fig

# Predefined recommendation rules - simplified from your original
RECOMMENDATION_RULES = {
    '0': {
        'primary': [
            "Schedule annual well-woman exam",
            "Consider mental health screening"
        ],
        'screenings': [
            "Pap smear every 3 years",
            "STI screening"
        ],
        'lifestyle': [
            "Stress management techniques",
            "Regular physical activity"
        ]
    },
    '1': {
        'primary': [
            "Monitor blood pressure regularly",
            "Schedule diabetes screening"
        ],
        'screenings': [
            "Mammogram",
            "Bone density test"
        ],
        'lifestyle': [
            "Heart-healthy diet",
            "Regular physical activity"
        ]
    },
    '2': {
        'primary': [
            "Regular reproductive health check-ups",
            "Fertility counseling if planning pregnancy"
        ],
        'screenings': [
            "Pap smear",
            "HPV testing"
        ],
        'lifestyle': [
            "Prenatal vitamins if planning pregnancy",
            "Balanced nutrition"
        ]
    },
    '3': {
        'primary': [
            "Chronic disease management",
            "Medication review"
        ],
        'screenings': [
            "Blood pressure monitoring",
            "A1C testing"
        ],
        'lifestyle': [
            "Mediterranean diet",
            "Low-impact exercise"
        ]
    },
    '4': {
        'primary': [
            "Preventive screening schedule",
            "Age-appropriate vaccinations"
        ],
        'screenings': [
            "Mammogram",
            "Colorectal cancer screening"
        ],
        'lifestyle': [
            "Weight management",
            "Bone-strengthening exercises"
        ]
    }
}

# Set page configuration with a friendly theme
st.set_page_config(
    page_title="Women's Wellness Coach",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for a more friendly appearance
st.markdown("""
<style>
    .main {
        background-color: #fcf7fa;
        padding: 20px;
    }
    .stApp header {
        background-color: #f9e6f0;
    }
    .stButton button {
        background-color: #e75480;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #e75480;
    }
    .stProgress .st-bo {
        background-color: #e75480;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if not exist
if 'platform' not in st.session_state:
    st.session_state.platform = MultiModalHealthPlatform()
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = f"P{np.random.randint(1000, 9999)}"
if 'data_submitted' not in st.session_state:
    st.session_state.data_submitted = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'profile_viz' not in st.session_state:
    st.session_state.profile_viz = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0
    
# Initialize all form fields with default values to prevent errors
default_fields = {
    'age_group': "18-30",
    'health_condition': "None",
    'bmi': 25.0,
    'systolic_bp': 120,
    'diastolic_bp': 80,
    'heart_rate': 75,
    'clinical_notes': "",
    'activity_level': "Moderate",
    'sleep_quality': "Good",
    'care_preference': "Digital (virtual visits)",
    'health_priority': "General wellness"
}

for field, value in default_fields.items():
    if field not in st.session_state:
        st.session_state[field] = value

# Function to update current page
def navigate_to(page_name):
    # Update progress based on page
    progress_map = {
        "Welcome": 0.0,
        "Health Basics": 0.25,
        "Lifestyle": 0.5,
        "Preferences": 0.75,
        "My Recommendations": 1.0 if st.session_state.data_submitted else 0.75
    }
    st.session_state.progress = progress_map[page_name]
    
    # Set the current page
    st.session_state.current_page = page_name
    
    # Use st.experimental_rerun() to refresh the page
    st.rerun()

# Initialize current page if not exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Welcome"

# Create friendly sidebar for navigation
with st.sidebar:
    st.title("💗 My Wellness Journey")
    st.markdown("---")
    
    # Display progress
    if st.session_state.progress > 0:
        st.progress(st.session_state.progress)
        st.markdown(f"Profile completion: {int(st.session_state.progress * 100)}%")
    
    # Navigation
    selected_page = st.radio(
        "Your journey steps:",
        ["Welcome", "Health Basics", "Lifestyle", "Preferences", "My Recommendations"],
        index=["Welcome", "Health Basics", "Lifestyle", "Preferences", "My Recommendations"].index(st.session_state.current_page)
    )
    
    # If page selection changes in sidebar, update the current page
    if selected_page != st.session_state.current_page:
        navigate_to(selected_page)
    
    st.markdown("---")
    st.info("Your information is private and secure. We never share your data without permission.")

# Function to process user data and generate recommendations - simplified
def create_personalized_recommendations():
    platform = st.session_state.platform
    
    # Gather all health data
    health_data = {
        'age_group': st.session_state.age_group,
        'bmi': st.session_state.bmi,
        'blood_pressure': f"{st.session_state.systolic_bp}/{st.session_state.diastolic_bp}",
        'heart_rate': st.session_state.heart_rate,
        'health_condition': st.session_state.health_condition
    }
    
    clinical_notes = st.session_state.clinical_notes
    
    # Wearable data
    wearable_data = {
        'activity_level': st.session_state.activity_level,
        'sleep_quality': st.session_state.sleep_quality
    }
    
    # Preferences
    preferences = {
        'care_preference': st.session_state.care_preference,
        'health_priority': st.session_state.health_priority
    }
    
    # Process data (simplified)
    integrated_data = platform.preprocess_data(
        health_data, clinical_notes, wearable_data, preferences
    )
    
    # Generate cluster assignment
    cluster_id = platform.generate_clusters(integrated_data)
    
    # Generate recommendations
    recommendations = platform.generate_health_recommendations(
        st.session_state.patient_id, cluster_id
    )
    
    # Create visualization
    profile_viz = platform.visualize_health_profile(cluster_id)
    
    # Store in session state
    st.session_state.recommendations = recommendations
    st.session_state.profile_viz = profile_viz
    st.session_state.data_submitted = True
    st.session_state.progress = 1.0  # Complete

# Display different pages based on navigation
if st.session_state.current_page == "Welcome":
    st.header("Welcome to Your Personal Wellness Coach")
    
    # Welcoming card with clear instructions
    st.markdown("""
    <div class="card">
    <h3>Your path to better health starts here</h3>
    <p>This wellness coach uses smart technology to provide personalized health guidance just for you. 
    We'll help you understand your health and make choices that work for your unique needs.</p>
    
    <h4>How it works:</h4>
    <ol>
        <li>Answer a few questions about your health and preferences</li>
        <li>Our system analyzes your information</li>
        <li>Get personalized recommendations for your wellness journey</li>
    </ol>
    
    <p>Ready to start? Click <b>Health Basics</b> in the sidebar or the button below to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start My Journey →", use_container_width=True):
        navigate_to("Health Basics")
    
    # Example of insights (simplified)
    st.subheader("What You'll Discover")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card" style="text-align: center;">
        <h3>✓ Health Insights</h3>
        <p>Understand your unique health patterns and priorities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center;">
        <h3>🩺 Screening Guide</h3>
        <p>Know which health screenings are right for you.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="card" style="text-align: center;">
        <h3>🥗 Lifestyle Tips</h3>
        <p>Simple, practical wellness advice for your daily life.</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.current_page == "Health Basics":
    st.header("Tell Us About Your Health")
    
    # Basic health information in a friendly card format
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Age and basic health
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.age_group = st.selectbox(
            "Your age range", 
            options=["18-30", "31-45", "46-60", "61+"]
        )
    
    with col2:
        st.session_state.health_condition = st.selectbox(
            "Do you have any ongoing health conditions?",
            options=["None", "Anxiety", "Depression", "Hypertension", "Diabetes", "PCOS", 
                    "Osteoporosis", "Heart condition", "Other"]
        )
    
    # Vital signs with helpful descriptions
    st.subheader("Your health numbers")
    st.markdown("These help us understand your baseline health status.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.bmi = st.number_input(
            "BMI (Body Mass Index)", 
            min_value=15.0, 
            max_value=45.0, 
            value=st.session_state.bmi,
            step=0.1,
            format="%.1f",
            help="BMI is a measure of body fat based on height and weight."
        )
    
    with col2:
        st.session_state.systolic_bp = st.number_input(
            "Systolic Blood Pressure (top number)", 
            min_value=80, 
            max_value=200, 
            value=st.session_state.systolic_bp,
            help="The pressure when your heart pushes blood out."
        )
        st.session_state.diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (bottom number)", 
            min_value=40, 
            max_value=120, 
            value=st.session_state.diastolic_bp,
            help="The pressure when your heart rests between beats."
        )
    
    with col3:
        st.session_state.heart_rate = st.number_input(
            "Resting Heart Rate (beats per minute)", 
            min_value=40, 
            max_value=120, 
            value=st.session_state.heart_rate,
            help="Your heart rate when you're relaxed and sitting still."
        )
    
    # Health concerns in simple language
    st.subheader("Any health concerns you want to share?")
    st.session_state.clinical_notes = st.text_area(
        "Tell us in your own words about any health concerns or questions (optional)",
        st.session_state.clinical_notes,
        height=100,
        help="This helps us understand your unique situation better."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Next: Lifestyle →", use_container_width=True):
            navigate_to("Lifestyle")

elif st.session_state.current_page == "Lifestyle":
    st.header("Your Daily Health Habits")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("""
    How active you are and how well you sleep greatly affect your overall health.
    Tell us about your typical patterns:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Physical Activity")
        st.session_state.activity_level = st.select_slider(
            "How would you describe your activity level?",
            options=["Very Low", "Low", "Moderate", "High", "Very High"],
            value=st.session_state.activity_level,
            help="Very Low: Mostly sitting, Low: Light activities, Moderate: Regular walking, High: Regular exercise, Very High: Intense workouts"
        )
        
        # Simple visual representation of activity level
        activity_levels = {
            "Very Low": "🧍‍♀️",
            "Low": "🚶‍♀️",
            "Moderate": "🚶‍♀️🚶‍♀️",
            "High": "🏃‍♀️🏃‍♀️",
            "Very High": "🏃‍♀️🏃‍♀️🏃‍♀️"
        }
        
        st.markdown(f"<h3 style='text-align: center; font-size: 30px;'>{activity_levels[st.session_state.activity_level]}</h3>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("Sleep Quality")
        st.session_state.sleep_quality = st.select_slider(
            "How would you rate your typical sleep quality?",
            options=["Poor", "Fair", "Good", "Excellent"],
            value=st.session_state.sleep_quality,
            help="Poor: Trouble falling/staying asleep, Fair: Occasional issues, Good: Regular restful sleep, Excellent: Consistently refreshed"
        )
        
        # Simple visual representation of sleep quality
        sleep_levels = {
            "Poor": "😴💭",
            "Fair": "😴",
            "Good": "😴💤",
            "Excellent": "😴💤💤"
        }
        
        st.markdown(f"<h3 style='text-align: center; font-size: 30px;'>{sleep_levels[st.session_state.sleep_quality]}</h3>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            navigate_to("Health Basics")
    with col2:
        if st.button("Next: Preferences →", use_container_width=True):
            navigate_to("Preferences")

elif st.session_state.current_page == "Preferences":
    st.header("Your Health Preferences")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("""
    Everyone's health journey is different. Tell us what matters most to you.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.care_preference = st.selectbox(
            "How do you prefer to receive healthcare?",
            options=["Digital (virtual visits)", "In-person care", "Combination of both"],
            index=["Digital (virtual visits)", "In-person care", "Combination of both"].index(st.session_state.care_preference),
            help="This helps us tailor recommendations to your comfort level"
        )
    
    with col2:
        priorities = ["Mental wellbeing", "Preventive health", "Reproductive health", 
                    "Physical fitness", "Managing a condition", "Menopause support", 
                    "General wellness"]
        st.session_state.health_priority = st.selectbox(
            "What's your main health focus right now?",
            options=priorities,
            index=priorities.index(st.session_state.health_priority),
            help="We'll prioritize recommendations for this area"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Submit button for all data with clear call to action
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            navigate_to("Lifestyle")
    with col2:
        if st.button("Create My Wellness Plan →", type="primary", use_container_width=True):
            with st.spinner("Creating your personalized wellness plan..."):
                create_personalized_recommendations()
            st.balloons()
            navigate_to("My Recommendations")

elif st.session_state.current_page == "My Recommendations":
    st.header("Your Personal Wellness Plan")
    
    if not st.session_state.data_submitted:
        st.warning("Please complete all the previous steps to see your personalized recommendations.")
        st.markdown("""
        <div class="card">
        <p>Start your wellness journey by clicking on <b>Health Basics</b> in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Health Basics →"):
            navigate_to("Health Basics")
    else:
        recommendations = st.session_state.recommendations
        
        # Display recommendations in an attractive format
        if recommendations:
            # Health profile visualization
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Your Health Profile")
            st.pyplot(st.session_state.profile_viz)
            
            # Recommendations in three columns with icons
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🩺 Health Checkups")
                for rec in recommendations['primary']:
                    st.markdown(f"• {rec}")
            
            with col2:
                st.markdown("### 🔍 Screenings")
                for rec in recommendations['screenings']:
                    st.markdown(f"• {rec}")
            
            with col3:
                st.markdown("### 🥗 Daily Habits")
                for rec in recommendations['lifestyle']:
                    st.markdown(f"• {rec}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Next steps
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Next Steps")
            st.markdown("""
            1. **Save these recommendations** for your next doctor's visit
            2. **Start with small changes** to your daily habits
            3. **Return monthly** to update your profile and get new insights
            """)
            
            # Share options
            if st.button("💌 Email My Recommendations"):
                st.success("In a real app, this would email your recommendations to you.")
            
            # Back button to adjust responses
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("← Start Over", use_container_width=True):
                    # Reset the data but keep the user info
                    st.session_state.data_submitted = False
                    st.session_state.recommendations = None
                    st.session_state.profile_viz = None
                    navigate_to("Health Basics")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.error("Something went wrong. Please try creating your recommendations again.")

# Simplified footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Women's Wellness Coach | Your Health Journey, Your Way</p>
</div>
""", unsafe_allow_html=True)