import streamlit as st
import pandas as pd
import os
import numpy as np
from google import genai
import json
import networkx as nx
import matplotlib.pyplot as plt
import altair as alt
from matplotlib.figure import Figure
from dotenv import load_dotenv
from datetime import datetime
from chatbot import (
    load_datasets, 
    GraphBuilderAgent, 
    ChatbotAgent, 
    ReportGeneratorAgent,
    sanitize_text
)
import httpx

# Simple monkey patch for httpx to handle non-ASCII characters in headers
original_normalize = httpx._models._normalize_header_value

def safe_normalize_header_value(value, encoding=None):
    """Ensure header values are ASCII-compatible by removing problematic characters"""
    if isinstance(value, str):
        # Replace the specific problematic character that's causing the error
        value = value.replace('\u2028', ' ')
        # Convert to ASCII, ignoring any remaining non-ASCII characters
        value = value.encode('ascii', 'ignore').decode('ascii')
    return original_normalize(value, encoding)

# Apply the monkey patch
httpx._models._normalize_header_value = safe_normalize_header_value
print("✓ HTTPX headers patched for ASCII compatibility")

# Add a custom write function that sanitizes text
def safe_write(obj, *args, **kwargs):
    """Use this instead of st.write to ensure ASCII compatibility"""
    if isinstance(obj, str):
        obj = sanitize_text(obj)
    st.write(obj, *args, **kwargs)
    
# Add a custom markdown function that sanitizes text
def safe_markdown(text, *args, **kwargs):
    """Use this instead of st.markdown to ensure ASCII compatibility"""
    if isinstance(text, str):
        text = sanitize_text(text)
    st.markdown(text, *args, **kwargs)

# Load environment variables from .env file
load_dotenv()

# Get API key using our robust function
api_key = os.getenv("GEMINI_API_KEY").strip()
if not api_key:
   api_key = os.environ.get("GEMINI_API_KEY").strip()
# Create the client
model = genai.Client(api_key=api_key)

# App title and description
st.title("HR Analytics Assistant")
st.markdown("Upload employee data, analyze issues, and generate comprehensive reports.")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'employee_id' not in st.session_state:
    st.session_state.employee_id = None
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None
if 'issues' not in st.session_state:
    st.session_state.issues = []
if 'report' not in st.session_state:
    st.session_state.report = None
if 'datasets_loaded' not in st.session_state:
    st.session_state.datasets_loaded = False
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = {}
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Chat"

# Sidebar for data upload and employee selection
with st.sidebar:
    st.header("Data Management")
    
    # Option to use sample data or upload own data
    data_option = st.radio("Choose data source:", ["Use sample data", "Upload your own data"])
    
    if data_option == "Upload your own data":
        st.write("Upload your CSV files:")
        activity_file = st.file_uploader("Activity Tracker", type="csv")
        leave_file = st.file_uploader("Leave Data", type="csv")
        onboarding_file = st.file_uploader("Onboarding Data", type="csv")
        performance_file = st.file_uploader("Performance Data", type="csv")
        rewards_file = st.file_uploader("Rewards Data", type="csv")
        vibemeter_file = st.file_uploader("Vibemeter Data", type="csv")
        
        if all([activity_file, leave_file, onboarding_file, performance_file, rewards_file, vibemeter_file]):
            try:
                # Load uploaded data
                activity_df = pd.read_csv(activity_file)
                leave_df = pd.read_csv(leave_file)
                onboarding_df = pd.read_csv(onboarding_file)
                performance_df = pd.read_csv(performance_file)
                rewards_df = pd.read_csv(rewards_file)
                vibemeter_df = pd.read_csv(vibemeter_file)
                st.session_state.datasets_loaded = True
                st.session_state.raw_data = {
                    'activity': activity_df,
                    'leave': leave_df,
                    'onboarding': onboarding_df,
                    'performance': performance_df,
                    'rewards': rewards_df,
                    'vibemeter': vibemeter_df
                }
                st.success("All datasets loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
    else:
        # Use sample data
        try:
            activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df = load_datasets()
            st.session_state.datasets_loaded = True
            st.session_state.raw_data = {
                'activity': activity_df,
                'leave': leave_df,
                'onboarding': onboarding_df,
                'performance': performance_df,
                'rewards': rewards_df,
                'vibemeter': vibemeter_df
            }
            st.success("Sample datasets loaded successfully!")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
    
    # Employee selection
    if st.session_state.datasets_loaded:
        # Get unique employee IDs from vibemeter data
        employee_ids = vibemeter_df['Employee_ID'].unique().tolist()
        selected_employee = st.selectbox("Select Employee ID:", employee_ids)
        
        if st.button("Analyze Employee"):
            with st.spinner("Building knowledge graph..."):
                # Create graph builder
                graph_builder = GraphBuilderAgent(
                    activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df
                )
                # Build knowledge graph
                knowledge_graph, issues = graph_builder.run(selected_employee)
                
                # Update session state
                st.session_state.employee_id = selected_employee
                st.session_state.knowledge_graph = knowledge_graph
                st.session_state.issues = issues
                st.session_state.chatbot = ChatbotAgent()
                st.session_state.conversation = []
                st.session_state.report = None
                
                st.success(f"Found {len(issues)} potential issues for Employee {selected_employee}")

# Main content area - tabs for different views
if st.session_state.employee_id and st.session_state.knowledge_graph:
    tabs = ["Chat", "Analytics Dashboard", "Report Builder", "Full Report"]
    
    # Use radio buttons for tabs instead of st.tabs() for better control
    st.session_state.selected_tab = st.radio("Select View:", tabs, 
                                           index=tabs.index(st.session_state.selected_tab) if st.session_state.selected_tab in tabs else 0,
                                           horizontal=True)
    
    st.header(f"Employee: {st.session_state.employee_id}")
    
    # Tab 1: Chat Interface
    if st.session_state.selected_tab == "Chat":
        with st.expander("Identified Issues", expanded=False):
            for issue in st.session_state.issues:
                st.write(f"**{issue['type']}** ({issue['severity']}): {issue['description']}")
        
        # Chat interface
        st.subheader("Chat with HR Assistant")
        
        # Initialize conversation if empty
        if not st.session_state.conversation:
            greeting = st.session_state.chatbot.start_conversation(st.session_state.issues)
            st.session_state.conversation.append({"role": "assistant", "content": greeting})
        
        # Display conversation history
        for message in st.session_state.conversation:
            if message["role"] == "assistant":
                st.markdown(f"**HR Assistant**: {message['content']}")
            else:
                st.markdown(f"**You**: {message['content']}")
        
        # User input
        user_input = st.text_input("Type your message:", key="user_message")
        
        if st.button("Send") and user_input:
            # Add user message to conversation
            st.session_state.conversation.append({"role": "user", "content": user_input})
            
            # Process response
            chatbot = st.session_state.chatbot
            
            # Analyze user response
            analysis = chatbot.analyze_response(user_input, st.session_state.conversation[-4:])
            
            # Update chatbot state
            if chatbot.current_issue_index < len(chatbot.current_issues):
                current_issue = chatbot.current_issues[chatbot.current_issue_index]
                sufficient_depth = analysis.get("SUFFICIENT_DEPTH", "no").lower() == "yes"
                
                if sufficient_depth or chatbot.follow_up_count >= chatbot.max_follow_ups:
                    # Mark current issue as explored
                    if current_issue:
                        chatbot.explored_issues[current_issue['type']]['explored'] = True
                    
                    # Move to next issue
                    chatbot.current_issue_index += 1
                    chatbot.follow_up_count = 0
                else:
                    # Continue with follow-up
                    chatbot.follow_up_count += 1
            
            # Check if all issues have been explored
            all_explored = all(data['explored'] for data in chatbot.explored_issues.values())
            
            # Generate next message
            if all_explored or chatbot.current_issue_index >= len(chatbot.current_issues):
                # Generate solutions
                solutions = chatbot.generate_solution_summary()
                closing_message = f"Thank you for sharing your thoughts. Based on our conversation, here are some suggestions:\n\n{solutions}"
                st.session_state.conversation.append({"role": "assistant", "content": closing_message})
            else:
                # Generate next question
                next_question = chatbot.generate_question(st.session_state.conversation[-6:])
                st.session_state.conversation.append({"role": "assistant", "content": next_question})
            
            # Force refresh to display new messages
            st.rerun()
    
    # Tab 2: Analytics Dashboard
    elif st.session_state.selected_tab == "Analytics Dashboard":
        st.subheader("Employee Analytics Dashboard")
        
        # Extract metrics from knowledge graph
        metrics = {}
        graph = st.session_state.knowledge_graph
        employee_id = st.session_state.employee_id
        
        # Function to extract metrics
        def extract_metrics():
            metrics = {}
            
            # Extract vibe metrics
            if f"{employee_id}_vibe" in graph.nodes:
                vibe_node = graph.nodes[f"{employee_id}_vibe"]
                vibe_scores = vibe_node.get('scores', [])
                metrics['vibe_scores'] = vibe_scores
                metrics['vibe_trend'] = vibe_node.get('trend', 'unknown')
            
            # Extract activity metrics
            if f"{employee_id}_activity" in graph.nodes:
                activity_node = graph.nodes[f"{employee_id}_activity"]
                metrics['avg_work_hours'] = activity_node.get('avg_work_hours', 0)
                metrics['avg_meetings'] = activity_node.get('avg_meetings', 0)
                metrics['avg_messages'] = activity_node.get('avg_messages', 0)
                metrics['avg_emails'] = activity_node.get('avg_emails', 0)
            
            # Extract performance metrics
            if f"{employee_id}_performance" in graph.nodes:
                performance_node = graph.nodes[f"{employee_id}_performance"]
                metrics['performance_rating'] = performance_node.get('rating', 0)
                metrics['promotion_consideration'] = performance_node.get('promotion', 'unknown')
            
            # Extract leave metrics
            if f"{employee_id}_leave" in graph.nodes:
                leave_node = graph.nodes[f"{employee_id}_leave"]
                metrics['leave_count'] = leave_node.get('leave_count', 0)
                metrics['leave_days_total'] = leave_node.get('leave_days_total', 0)
                metrics['leave_types'] = leave_node.get('leave_types', {})
            
            # Extract rewards metrics
            if f"{employee_id}_rewards" in graph.nodes:
                rewards_node = graph.nodes[f"{employee_id}_rewards"]
                metrics['reward_count'] = rewards_node.get('reward_count', 0)
                metrics['rewards_points'] = rewards_node.get('rewards_points', 0)
            
            return metrics
        
        # Get employee metrics
        employee_metrics = extract_metrics()
        
        # Create metrics visualization
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Average vibe score with emoji based on value
            if 'vibe_scores' in employee_metrics:
                avg_vibe = sum(employee_metrics['vibe_scores'])/len(employee_metrics['vibe_scores']) if employee_metrics['vibe_scores'] else 0
                emoji = "😀" if avg_vibe >= 7 else "🙂" if avg_vibe >= 5 else "😐" if avg_vibe >= 3 else "😟"
                st.metric("Average Vibe", f"{avg_vibe:.1f} {emoji}")
            
        with col2:
            # Work hours with color based on value
            if 'avg_work_hours' in employee_metrics:
                hrs = employee_metrics['avg_work_hours']
                color_indicator = "normal" if hrs <= 8 else "inverse" if hrs <= 9 else "inverse"
                st.metric("Avg Work Hours", f"{hrs:.1f}", delta_color=color_indicator)
        
        with col3:
            # Performance rating with indicator
            if 'performance_rating' in employee_metrics:
                rating = employee_metrics['performance_rating']
                st.metric("Performance", f"{rating}/5")
        
        with col4:
            # Leave days
            if 'leave_days_total' in employee_metrics:
                st.metric("Total Leave Days", employee_metrics['leave_days_total'])
        
        # Create dashboard sections with visualizations
        st.markdown("### Vibe Trend Analysis")
        
        # Vibe trend visualization - REPLACED PLOTLY WITH ALTAIR
        if 'vibe_scores' in employee_metrics:
            # Get vibe data
            employee_vibes = st.session_state.raw_data['vibemeter']
            employee_vibes = employee_vibes[employee_vibes['Employee_ID'] == employee_id]
            
            if not employee_vibes.empty:
                # Convert to datetime if needed
                if 'Response_Date' in employee_vibes.columns:
                    if not pd.api.types.is_datetime64_any_dtype(employee_vibes['Response_Date']):
                        employee_vibes['Response_Date'] = pd.to_datetime(employee_vibes['Response_Date'], errors='coerce')
                
                # Sort by date
                employee_vibes = employee_vibes.sort_values(by='Response_Date')
                
                # Create vibe trend chart with Altair
                chart = alt.Chart(employee_vibes).mark_line(point=True).encode(
                    x=alt.X('Response_Date:T', title='Date'),
                    y=alt.Y('Vibe_Score:Q', title='Vibe Score', scale=alt.Scale(domain=[0, 10])),
                    tooltip=['Response_Date:T', 'Vibe_Score:Q']
                ).properties(
                    title='Vibe Score Trend Over Time',
                    height=300
                )
                
                # Add a reference line at score 5
                reference_line = alt.Chart(pd.DataFrame({'y': [5]})).mark_rule(
                    strokeDash=[6, 6],
                    color='gray'
                ).encode(y='y')
                
                # Combine the two charts
                chart = alt.layer(chart, reference_line)
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No vibe data available for this employee")
        
        # Work patterns visualization
        st.markdown("### Work Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            # Get activity data for this employee
            if 'avg_work_hours' in employee_metrics:
                # REPLACED PLOTLY GAUGE WITH MATPLOTLIB
                # Create a gauge chart for work hours using matplotlib
                fig, ax = plt.subplots(figsize=(4, 3))
                
                # Define the gauge chart
                hrs = employee_metrics['avg_work_hours']
                gauge_angle = 180 * (hrs / 12)  # Convert to angle (max 12 hours)
                
                # Draw the gauge background
                ax.add_patch(plt.matplotlib.patches.Wedge(
                    center=(0.5, 0), 
                    r=0.8, 
                    theta1=0, 
                    theta2=180, 
                    fc='lightgray',
                    alpha=0.3
                ))
                
                # Color coding the gauge
                ax.add_patch(plt.matplotlib.patches.Wedge(
                    center=(0.5, 0), 
                    r=0.8, 
                    theta1=0, 
                    theta2=180*(8/12), 
                    fc='lightgreen',
                    alpha=0.5
                ))
                
                ax.add_patch(plt.matplotlib.patches.Wedge(
                    center=(0.5, 0), 
                    r=0.8, 
                    theta1=180*(8/12), 
                    theta2=180*(9/12), 
                    fc='khaki',
                    alpha=0.5
                ))
                
                ax.add_patch(plt.matplotlib.patches.Wedge(
                    center=(0.5, 0), 
                    r=0.8, 
                    theta1=180*(9/12), 
                    theta2=180, 
                    fc='salmon',
                    alpha=0.5
                ))
                
                # Draw the needle
                ax.add_patch(plt.matplotlib.patches.Wedge(
                    center=(0.5, 0), 
                    r=0.05, 
                    theta1=0, 
                    theta2=360, 
                    fc='black'
                ))
                
                # Draw needle line
                ax.plot([0.5, 0.5 + 0.7 * np.cos(np.radians(180 - gauge_angle))],
                        [0, 0.7 * np.sin(np.radians(180 - gauge_angle))],
                        color='black', linewidth=2)
                
                # Add gauge labels
                ax.text(0.1, 0.15, '0', fontsize=12)
                ax.text(0.5, 0.8, f'{hrs:.1f}h', fontsize=14, ha='center')
                ax.text(0.9, 0.15, '12', fontsize=12)
                
                # Remove axis ticks and labels
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.1, 1)
                ax.axis('off')
                plt.title('Average Work Hours')
                
                st.pyplot(fig)
        
        with col2:
            # Workload distribution - REPLACED PLOTLY WITH ALTAIR
            if all(key in employee_metrics for key in ['avg_meetings', 'avg_messages', 'avg_emails']):
                # Create activity breakdown chart with Altair
                activity_data = pd.DataFrame({
                    'Activity': ["Meetings", "Emails", "Messages"],
                    'Count': [
                        employee_metrics.get('avg_meetings', 0), 
                        employee_metrics.get('avg_emails', 0), 
                        employee_metrics.get('avg_messages', 0)
                    ]
                })
                
                chart = alt.Chart(activity_data).mark_bar().encode(
                    x=alt.X('Activity:N', title='Activity Type'),
                    y=alt.Y('Count:Q', title='Average Daily Count'),
                    color=alt.Color('Count:Q', scale=alt.Scale(scheme='viridis'))
                ).properties(
                    title='Daily Communication Activity',
                    height=250
                )
                
                st.altair_chart(chart, use_container_width=True)
        
        # Leave pattern analysis - REPLACED PLOTLY WITH MATPLOTLIB
        if 'leave_types' in employee_metrics:
            st.markdown("### Leave Pattern Analysis")
            
            leave_types = employee_metrics['leave_types']
            if leave_types:
                # Create leave distribution chart using matplotlib
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = list(leave_types.keys())
                values = list(leave_types.values())
                
                # Create pie chart with matplotlib
                wedges, texts, autotexts = ax.pie(
                    values, 
                    labels=None,  # We'll add custom legend instead
                    autopct='%1.1f%%', 
                    startangle=90, 
                    wedgeprops={'width': 0.5, 'edgecolor': 'white'}
                )
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')  
                plt.title('Leave Distribution by Type')
                
                # Add a legend
                plt.legend(
                    wedges, 
                    labels,
                    title="Leave Types",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1)
                )
                
                st.pyplot(fig)
        
        # Issues breakdown
        st.markdown("### Identified Issues Analysis")
        
        # Categorize issues by type and severity
        issue_types = {}
        issue_severity = {'high': 0, 'medium': 0, 'low': 0}
        
        for issue in st.session_state.issues:
            issue_type = issue['type']
            severity = issue['severity']
            
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1
            issue_severity[severity] += 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create issue type distribution chart - REPLACED PLOTLY WITH ALTAIR
            if issue_types:
                issue_data = pd.DataFrame({
                    'Issue Type': list(issue_types.keys()),
                    'Count': list(issue_types.values())
                })
                
                chart = alt.Chart(issue_data).mark_bar().encode(
                    x=alt.X('Issue Type:N', title='Issue Category'),
                    y=alt.Y('Count:Q', title='Count'),
                    color=alt.Color('Issue Type:N', scale=alt.Scale(scheme='category10'))
                ).properties(
                    title='Issues by Category',
                    height=300
                )
                
                st.altair_chart(chart, use_container_width=True)
        
        with col2:
            # Create severity distribution chart - REPLACED PLOTLY WITH MATPLOTLIB
            fig, ax = plt.subplots(figsize=(5, 4))
            labels = list(issue_severity.keys())
            sizes = list(issue_severity.values())
            colors = ['red', 'orange', 'green']
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%', 
                startangle=90
            )
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')  
            plt.title('Issues by Severity')
            
            st.pyplot(fig)
    
    # Tab 3: Report Builder
    elif st.session_state.selected_tab == "Report Builder":
        st.subheader("Custom Report Builder")
        
        # Report sections selection
        st.markdown("### Select Report Sections")
        
        report_sections = {
            "executive_summary": st.checkbox("Executive Summary", value=True),
            "key_metrics": st.checkbox("Key Metrics", value=True),
            "identified_issues": st.checkbox("Identified Issues", value=True),
            "problem_categories": st.checkbox("Problem Categories", value=True),
            "root_causes": st.checkbox("Root Causes", value=True),
            "recommended_actions": st.checkbox("Recommended Actions", value=True),
            "criticality_assessment": st.checkbox("Mental Health Assessment", value=True),
        }
        
        # Issue filtering options
        st.markdown("### Filter Issues by Severity")
        severity_filter = st.multiselect(
            "Include issues with severity:",
            ["high", "medium", "low"],
            default=["high", "medium", "low"]
        )
        
        # Custom notes addition
        st.markdown("### Add Custom Notes")
        custom_notes = st.text_area(
            "Additional notes or observations:",
            placeholder="Enter any additional observations or context that should be included in the report..."
        )
        
        # Generate custom report button
        if st.button("Generate Custom Report"):
            with st.spinner("Building customized report..."):
                # Create report generator 
                report_generator = ReportGeneratorAgent()
                
                # Filter issues by severity if needed
                filtered_issues = [issue for issue in st.session_state.issues 
                                  if issue['severity'] in severity_filter]
                
                # Generate base report
                report = report_generator.run(
                    st.session_state.employee_id,
                    st.session_state.knowledge_graph,
                    filtered_issues,
                    st.session_state.conversation
                )
                
                # Customize the report based on selections
                custom_report = {}
                for section, include in report_sections.items():
                    if include and section in report:
                        custom_report[section] = report[section]
                
                # Add custom notes if provided
                if custom_notes:
                    custom_report["custom_notes"] = custom_notes
                
                # Add report generation metadata
                custom_report["report_metadata"] = {
                    "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "custom_report": True,
                    "included_sections": [section for section, include in report_sections.items() if include],
                    "severity_filter": severity_filter
                }
                
                # Save the custom report
                st.session_state.report = custom_report
                
                # Save to file
                custom_report_filename = f"emp_{st.session_state.employee_id}_custom_report.json"
                with open(custom_report_filename, "w") as f:
                    json.dump(custom_report, f, indent=2)
                
                st.success(f"Custom report generated and saved as {custom_report_filename}")
                
                # Automatically switch to the Full Report tab to view the report
                st.session_state.selected_tab = "Full Report"
                st.rerun()
    
    # Tab 4: Full Report View
    elif st.session_state.selected_tab == "Full Report":
        st.subheader("Employee Report")
        
        # Check if we have a report
        if not st.session_state.report:
            if st.button("Generate Full Report"):
                with st.spinner("Generating comprehensive report..."):
                    report_generator = ReportGeneratorAgent()
                    report = report_generator.run(
                        st.session_state.employee_id,
                        st.session_state.knowledge_graph,
                        st.session_state.issues,
                        st.session_state.conversation
                    )
                    st.session_state.report = report
                    st.success("Report generated successfully!")
                    st.rerun()
            else:
                st.info("Please generate a report first by clicking the button above or using the Report Builder tab.")
        else:
            # Display report in a more readable format
            report = st.session_state.report
            
            # Check for and display criticality warning first
            if "criticality" in report:
                criticality = report["criticality"]
                risk_level = criticality.get("risk_level", "unknown")
                urgent_action = criticality.get("urgent_action_required", False)
                
                # Show warning for medium or higher risk
                if risk_level in ["medium", "high", "critical"] or urgent_action:
                    st.error(f"⚠️ MENTAL HEALTH RISK LEVEL: {risk_level.upper()}")
                    st.warning("**Urgent action may be required**" if urgent_action else "**Attention recommended**")
                    
                    with st.expander("Mental Health Assessment Details", expanded=True):
                        st.write("**Indicators:**")
                        for indicator in criticality.get("indicators", []):
                            st.write(f"- {indicator}")
                        
                        st.write("**Recommendations:**")
                        for rec in criticality.get("recommendations", []):
                            st.write(f"- {rec}")
            
            # Executive Summary
            if "executive_summary" in report:
                st.markdown("### Executive Summary")
                st.write(report["executive_summary"])
            
            # Key Metrics
            if "key_metrics" in report:
                st.markdown("### Key Metrics")
                metrics = report["key_metrics"]
                
                # Create a cleaner metrics display with columns
                cols = st.columns(3)
                i = 0
                for metric, value in metrics.items():
                    # Skip nested dictionaries for this simple view
                    if not isinstance(value, dict):
                        cols[i % 3].metric(
                            label=metric.replace('_', ' ').title(),
                            value=value
                        )
                        i += 1
            
            # Problem Categories
            if "problem_categories" in report:
                st.markdown("### Problem Categories")
                categories = report["problem_categories"]
                
                # Create a table for categories
                category_data = []
                for category, details in categories.items():
                    score = details.get('score', 'N/A')
                    factors = ', '.join(details.get('factors', []))
                    evidence = details.get('evidence', 'No evidence provided')
                    category_data.append({
                        "Category": category,
                        "Relevance Score": score,
                        "Key Factors": factors,
                        "Evidence": evidence
                    })
                
                if category_data:
                    category_df = pd.DataFrame(category_data)
                    st.dataframe(category_df, use_container_width=True)
            
            # Identified Issues
            if "identified_issues" in report:
                st.markdown("### Identified Issues")
                
                for issue in report["identified_issues"]:
                    severity = issue.get("severity", "medium")
                    color = "red" if severity == "high" else "orange" if severity == "medium" else "green"
                    
                    st.markdown(
                        f"<div style='padding:10px; border-left:4px solid {color}; margin-bottom:10px;'>"
                        f"<strong>{issue.get('type', 'Issue').title()}</strong> ({severity})<br>"
                        f"{issue.get('description', 'No description')}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            
            # Root Causes
            if "root_causes" in report:
                st.markdown("### Root Causes")
                
                if isinstance(report["root_causes"], list):
                    for cause in report["root_causes"]:
                        st.write(f"- {cause}")
                elif isinstance(report["root_causes"], dict):
                    for issue_type, causes in report["root_causes"].items():
                        st.markdown(f"**{issue_type.title()}**")
                        for cause in causes:
                            st.write(f"- {cause}")
            
            # Recommended Actions
            if "recommended_actions" in report:
                st.markdown("### Recommended Actions")
                
                for i, action in enumerate(report["recommended_actions"]):
                    st.markdown(f"**{i+1}.** {action}")
            
            # Custom notes if present
            if "custom_notes" in report:
                st.markdown("### Additional Notes")
                st.write(report["custom_notes"])
            
            # Download button
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="Download Report (JSON)",
                data=report_json,
                file_name=f"employee_{st.session_state.employee_id}_report.json",
                mime="application/json"
            )
            
            # Option to generate PDF version
            st.markdown("---")
            st.markdown("### Export Options")
            if st.button("Export as PDF Report"):
                st.warning("PDF export functionality would be implemented here. This would convert the JSON report to a formatted PDF document.")
                # In a real implementation, this would use a library like ReportLab or WeasyPrint
                # to generate a properly formatted PDF version of the report

else:
    st.info("Please select an employee from the sidebar to begin analysis.")