import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from google import genai 
from typing import List, Dict, Any, Tuple
from langgraph.graph import StateGraph, END
import os
import json
import numpy as np
import streamlit as st

# Simplified text sanitization - replaces multiple ASCII handling functions
def sanitize_text(text):
    """Clean text to ensure ASCII compatibility and handle common Unicode characters"""
    if not isinstance(text, str):
        return text
        
    # Replace common problematic Unicode characters
    replacements = {
        '\u2028': '\n',      # Line separator
        '\u2029': '\n\n',    # Paragraph separator
        '\u00A0': ' ',       # Non-breaking space
        '\u200B': '',        # Zero-width space
        '\u200E': '',        # Left-to-right mark
        '\u200F': '',        # Right-to-left mark
        '\u2018': "'", '\u2019': "'",  # Smart single quotes
        '\u201C': '"', '\u201D': '"',  # Smart double quotes
        '\u2013': '-', '\u2014': '--', # En and em dashes
        '\u2026': '...'      # Ellipsis
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Ensure ASCII only
    return text.encode('ascii', 'ignore').decode('ascii')

# Simple JSON serialization helper
def ensure_json_serializable(obj):
    """Convert non-serializable objects to JSON-serializable types"""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)

# Get API key from environment variables or Streamlit secrets
def get_api_key():
    """Get the API key from environment variables"""
    return os.getenv("GEMINI_API_KEY", "").strip()

# Define data directory path relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
print(f"Looking for data in: {DATA_DIR}")

# Check for data directory and set fallbacks if needed
if not os.path.isdir(DATA_DIR):
    print(f"WARNING: Data directory not found at {DATA_DIR}")
    fallback_locations = [
        "data",  # Try relative to current working directory
        os.path.join(os.path.dirname(SCRIPT_DIR), "data"),  # Try parent directory
        "/app/data"  # Try Hugging Face container path
    ]
    
    for location in fallback_locations:
        print(f"Trying fallback location: {location}")
        if os.path.isdir(location):
            DATA_DIR = location
            print(f"Found data directory at: {location}")
            break
else:
    print(f"✓ Data directory found at: {DATA_DIR}")
    print("Data directory contents:")
    try:
        for file in os.listdir(DATA_DIR):
            print(f" - {file}")
    except Exception as e:
        print(f"Error listing directory contents: {e}")

# Define relative paths to data files
ACTIVITY_PATH = os.path.join(DATA_DIR, "activity_tracker_dataset.csv")
LEAVE_PATH = os.path.join(DATA_DIR, "leave_dataset.csv")
ONBOARDING_PATH = os.path.join(DATA_DIR, "onboarding_dataset.csv")
PERFORMANCE_PATH = os.path.join(DATA_DIR, "performance_dataset.csv")
REWARDS_PATH = os.path.join(DATA_DIR, "rewards_dataset.csv")
VIBEMETER_PATH = os.path.join(DATA_DIR, "vibemeter_dataset.csv")

# List all paths for debugging
print("Data paths:")
for path_name, path in [
    ("Activity", ACTIVITY_PATH), 
    ("Leave", LEAVE_PATH),
    ("Onboarding", ONBOARDING_PATH),
    ("Performance", PERFORMANCE_PATH),
    ("Rewards", REWARDS_PATH),
    ("Vibemeter", VIBEMETER_PATH)
]:
    print(f"- {path_name}: {path} - {'EXISTS' if os.path.exists(path) else 'MISSING'}")

# Function to load datasets from CSV files
def load_datasets():
    """Load all datasets from CSV files"""
    try:
        # Load each dataset with error handling
        datasets = {}
        for name, path in [
            ("activity", ACTIVITY_PATH),
            ("leave", LEAVE_PATH),
            ("onboarding", ONBOARDING_PATH),
            ("performance", PERFORMANCE_PATH),
            ("rewards", REWARDS_PATH),
            ("vibemeter", VIBEMETER_PATH)
        ]:
            print(f"Loading {name} data...")
            datasets[name] = pd.read_csv(path)
            print(f"{name.capitalize()} data loaded successfully! Shape: {datasets[name].shape}")
        
        print("All datasets loaded successfully!")
        return (
            datasets["activity"], 
            datasets["leave"], 
            datasets["onboarding"], 
            datasets["performance"], 
            datasets["rewards"], 
            datasets["vibemeter"]
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Define the state schema for the workflow
class AgentState:
    def __init__(self):
        self.employee_id = None
        self.knowledge_graph = None
        self.issues = []
        self.conversation_history = []
        self.recommendations = []
        self.report = None
    
    def get(self, key, default=None):
        """Make the class dict-like for compatibility with StateGraph"""
        return getattr(self, key, default)

# Minimal wrapper for Gemini client that ensures ASCII-safe inputs
class SafeGeminiClient:
    def __init__(self, api_key):
        from google import genai
        self._client = genai.Client(api_key=api_key)
    
    def generate_content(self, contents, model="gemini-2.0-flash", **kwargs):
        """Generate content with sanitized inputs"""
        # Sanitize the input contents
        if isinstance(contents, str):
            contents = sanitize_text(contents)
            
        # Call the actual client
        return self._client.models.generate_content(
            contents=contents,
            model=model,
            **kwargs
        )
    
    # Forward all other methods to the underlying client
    def __getattr__(self, name):
        return getattr(self._client, name)

# 1. Graph Builder Agent - simplified and more modular
class GraphBuilderAgent:
    def __init__(self, activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df):
        self.datasets = {
            "activity": activity_df,
            "leave": leave_df,
            "onboarding": onboarding_df,
            "performance": performance_df,
            "rewards": rewards_df,
            "vibemeter": vibemeter_df
        }
        
    def run(self, employee_id):
        """Main entry point to build knowledge graph and identify issues"""
        knowledge_graph = self.build_knowledge_graph(employee_id)
        issues = self.identify_issues(knowledge_graph, employee_id)
        return knowledge_graph, issues
        
    def build_knowledge_graph(self, employee_id):
        """Build a knowledge graph for the employee with data from all datasets"""
        G = nx.Graph()
        
        # Add employee node
        G.add_node(employee_id, type='employee')
        
        # Add data from each dataset to the graph
        self._add_vibe_data(G, employee_id)
        self._add_activity_data(G, employee_id)
        self._add_leave_data(G, employee_id)
        self._add_performance_data(G, employee_id)
        self._add_rewards_data(G, employee_id)
        self._add_onboarding_data(G, employee_id)
        
        return G
    
    def _add_vibe_data(self, G, employee_id):
        """Add vibemeter data to the graph"""
        employee_vibes = self.datasets["vibemeter"][self.datasets["vibemeter"]['Employee_ID'] == employee_id]
        if employee_vibes.empty:
            return
            
        vibe_scores = list(employee_vibes['Vibe_Score'])
        vibe_dates = list(employee_vibes['Response_Date'])
        
        # Calculate vibe trend
        vibe_trend = 'stable'
        if len(vibe_scores) > 3:
            recent_vibes = vibe_scores[-3:]
            if all(recent_vibes[i] < recent_vibes[i-1] for i in range(1, len(recent_vibes))):
                vibe_trend = 'declining'
            elif all(recent_vibes[i] > recent_vibes[i-1] for i in range(1, len(recent_vibes))):
                vibe_trend = 'improving'
        
        # Add vibe node
        G.add_node(f"{employee_id}_vibe", type='vibe', scores=vibe_scores, trend=vibe_trend)
        G.add_edge(employee_id, f"{employee_id}_vibe", relation='has_vibe')
    
    def _add_activity_data(self, G, employee_id):
        """Add activity data to the graph"""
        employee_activity = self.datasets["activity"][self.datasets["activity"]['Employee_ID'] == employee_id]
        if employee_activity.empty:
            return
            
        recent_activities = employee_activity.sort_values('Date', ascending=False).head(10)
        avg_work_hours = recent_activities['Work_Hours'].mean()
        avg_messages = recent_activities['Teams_Messages_Sent'].mean()
        avg_emails = recent_activities['Emails_Sent'].mean()
        avg_meetings = recent_activities['Meetings_Attended'].mean()
        
        # Add activity node
        G.add_node(f"{employee_id}_activity", 
                  type='activity', 
                  avg_work_hours=avg_work_hours,
                  avg_messages=avg_messages,
                  avg_emails=avg_emails,
                  avg_meetings=avg_meetings)
        G.add_edge(employee_id, f"{employee_id}_activity", relation='has_activity')
    
    def _add_leave_data(self, G, employee_id):
        """Add leave data to the graph"""
        employee_leaves = self.datasets["leave"][self.datasets["leave"]['Employee_ID'] == employee_id]
        if employee_leaves.empty:
            return
            
        leave_count = len(employee_leaves)
        leave_days_total = employee_leaves['Leave_Days'].sum()
        leave_types = employee_leaves['Leave_Type'].value_counts().to_dict()
        
        # Add leave node
        G.add_node(f"{employee_id}_leave", 
                  type='leave', 
                  leave_count=leave_count,
                  leave_days_total=leave_days_total,
                  leave_types=leave_types)
        G.add_edge(employee_id, f"{employee_id}_leave", relation='has_leave')
    
    def _add_performance_data(self, G, employee_id):
        """Add performance data to the graph"""
        employee_performance = self.datasets["performance"][self.datasets["performance"]['Employee_ID'] == employee_id]
        if employee_performance.empty:
            return
            
        recent_performance = employee_performance.sort_values('Review_Period', ascending=False).iloc[0]
        rating = recent_performance['Performance_Rating']
        feedback = recent_performance['Manager_Feedback']
        promotion = recent_performance['Promotion_Consideration']
        
        # Add performance node
        G.add_node(f"{employee_id}_performance", 
                  type='performance', 
                  rating=rating,
                  feedback=feedback,
                  promotion=promotion)
        G.add_edge(employee_id, f"{employee_id}_performance", relation='has_performance')
    
    def _add_rewards_data(self, G, employee_id):
        """Add rewards data to the graph"""
        employee_rewards = self.datasets["rewards"][self.datasets["rewards"]['Employee_ID'] == employee_id]
        if employee_rewards.empty:
            return
            
        reward_count = len(employee_rewards)
        reward_types = employee_rewards['Award_Type'].value_counts().to_dict()
        rewards_points = employee_rewards['Reward_Points'].sum()
        
        # Add rewards node
        G.add_node(f"{employee_id}_rewards", 
                  type='rewards', 
                  reward_count=reward_count,
                  reward_types=reward_types,
                  rewards_points=rewards_points)
        G.add_edge(employee_id, f"{employee_id}_rewards", relation='has_rewards')
    
    def _add_onboarding_data(self, G, employee_id):
        """Add onboarding data to the graph"""
        employee_onboarding = self.datasets["onboarding"][self.datasets["onboarding"]['Employee_ID'] == employee_id]
        if employee_onboarding.empty:
            return
            
        onboarding_data = employee_onboarding.iloc[0]
        joining_date = onboarding_data['Joining_Date']
        feedback = onboarding_data['Onboarding_Feedback']
        mentor = onboarding_data['Mentor_Assigned']
        training = onboarding_data['Initial_Training_Completed']
        
        # Add onboarding node
        G.add_node(f"{employee_id}_onboarding", 
                  type='onboarding', 
                  joining_date=joining_date,
                  feedback=feedback,
                  mentor=mentor,
                  training=training)
        G.add_edge(employee_id, f"{employee_id}_onboarding", relation='has_onboarding')
    
    def identify_issues(self, G, employee_id):
        """Identify potential issues based on the knowledge graph"""
        issues = []
        
        # Check for various issues using helper methods
        self._check_vibe_issues(G, employee_id, issues)
        self._check_workload_issues(G, employee_id, issues)
        self._check_performance_issues(G, employee_id, issues)
        self._check_recognition_issues(G, employee_id, issues)
        self._check_leave_issues(G, employee_id, issues)
        self._check_onboarding_issues(G, employee_id, issues)
        
        return issues
    
    def _check_vibe_issues(self, G, employee_id, issues):
        """Check for vibe-related issues"""
        vibe_node_id = f"{employee_id}_vibe"
        if vibe_node_id not in G.nodes:
            return
            
        vibe_node = G.nodes[vibe_node_id]
        if vibe_node['trend'] == 'declining':
            issues.append({
                'type': 'vibe',
                'severity': 'high',
                'description': 'Declining vibe scores in recent surveys'
            })
        
        # Check for consistently low vibe scores
        vibe_scores = vibe_node['scores']
        if vibe_scores and sum(s < 5 for s in vibe_scores) / len(vibe_scores) > 0.5:
            issues.append({
                'type': 'vibe',
                'severity': 'high',
                'description': 'Consistently low vibe scores (below 5)'
            })
    
    def _check_workload_issues(self, G, employee_id, issues):
        """Check for workload-related issues"""
        activity_node_id = f"{employee_id}_activity"
        if activity_node_id not in G.nodes:
            return
            
        activity_node = G.nodes[activity_node_id]
        if activity_node['avg_work_hours'] > 9:
            issues.append({
                'type': 'workload',
                'severity': 'medium',
                'description': f'High average working hours: {activity_node["avg_work_hours"]:.1f} hours'
            })
        if activity_node['avg_meetings'] > 5:
            issues.append({
                'type': 'workload',
                'severity': 'medium',
                'description': f'High number of meetings: {activity_node["avg_meetings"]:.1f} per day'
            })
    
    def _check_performance_issues(self, G, employee_id, issues):
        """Check for performance-related issues"""
        performance_node_id = f"{employee_id}_performance"
        if performance_node_id not in G.nodes:
            return
            
        performance_node = G.nodes[performance_node_id]
        if performance_node['rating'] < 3:
            issues.append({
                'type': 'performance',
                'severity': 'high',
                'description': f'Low performance rating: {performance_node["rating"]}'
            })
        if performance_node['promotion'] == 'No':
            issues.append({
                'type': 'career',
                'severity': 'medium',
                'description': 'Not considered for promotion'
            })
    
    def _check_recognition_issues(self, G, employee_id, issues):
        """Check for recognition-related issues"""
        rewards_node_id = f"{employee_id}_rewards"
        performance_node_id = f"{employee_id}_performance"
        
        if rewards_node_id not in G.nodes or performance_node_id not in G.nodes:
            return
            
        rewards_node = G.nodes[rewards_node_id]
        performance_node = G.nodes[performance_node_id]
        
        if performance_node['rating'] >= 4 and rewards_node['reward_count'] == 0:
            issues.append({
                'type': 'recognition',
                'severity': 'medium',
                'description': 'High performer with no rewards'
            })
    
    def _check_leave_issues(self, G, employee_id, issues):
        """Check for leave-related issues"""
        leave_node_id = f"{employee_id}_leave"
        if leave_node_id not in G.nodes:
            return
            
        leave_node = G.nodes[leave_node_id]
        if leave_node['leave_count'] > 10:
            issues.append({
                'type': 'attendance',
                'severity': 'medium',
                'description': f'High leave count: {leave_node["leave_count"]} instances'
            })
        
        # Check for frequent sick leaves
        leave_types = leave_node.get('leave_types', {})
        sick_leaves = leave_types.get('Sick', 0)
        if sick_leaves > 5:
            issues.append({
                'type': 'health',
                'severity': 'medium',
                'description': f'Frequent sick leaves: {sick_leaves} instances'
            })
    
    def _check_onboarding_issues(self, G, employee_id, issues):
        """Check for onboarding-related issues"""
        onboarding_node_id = f"{employee_id}_onboarding"
        if onboarding_node_id not in G.nodes:
            return
            
        onboarding_node = G.nodes[onboarding_node_id]
        
        if onboarding_node['training'] == 'No':
            issues.append({
                'type': 'training',
                'severity': 'low',
                'description': 'Initial training not completed'
            })
            
        if 'negative' in str(onboarding_node['feedback']).lower():
            issues.append({
                'type': 'onboarding',
                'severity': 'medium',
                'description': 'Negative onboarding feedback'
            })

# Enhanced ChatbotAgent with intelligent question generation and response analysis
class ChatbotAgent:
    def __init__(self):
        self._model = None
        # Question templates for different issue types
        self.question_templates = {
            'vibe': "I noticed your recent vibe scores in our surveys show {issue_description}. What workplace factors might be affecting this?",
            'workload': "Your activity data shows {issue_description}. How is this affecting your work-life balance?",
            'performance': "In your last performance review, {issue_description}. What obstacles are you facing in meeting your goals?",
            'career': "According to your records, {issue_description}. What do you feel is limiting your career progression here?",
            'recognition': "Looking at your rewards history, {issue_description}. How does the recognition system affect your motivation?",
            'attendance': "Your leave records show {issue_description}. What workplace factors influence your attendance patterns?",
            'health': "Your sick leave data indicates {issue_description}. Are there workplace conditions affecting your wellbeing?",
            'training': "Records show {issue_description} regarding your initial training. What additional training would be valuable?",
            'onboarding': "Your onboarding records indicate {issue_description}. How could we have improved your initial experience?"
        }
        
        # Conversation tracking
        self.conversation = []
        self.current_issues = []
        self.explored_issues = {}
        self.current_issue_index = 0
        self.follow_up_count = 0
        self.max_follow_ups = 3
        
        # Analysis result tracking
        self.root_causes = {}
        self.themes = set()
        self.potential_solutions = {}
        self.sentiment_data = {}

    @property
    def model(self):
        """Lazily initialize and return the model client with current API key"""
        if self._model is None:
            api_key = get_api_key()
            if not api_key:
                raise ValueError("No valid Gemini API key found. Please add it to environment variables.")
            self._model = SafeGeminiClient(api_key=api_key)
        return self._model
        
    def start_conversation(self, issues):
        """Initialize a new conversation focused on discovering root causes and providing solutions"""
        self.conversation = []
        # Sort issues by severity
        self.current_issues = sorted(issues, key=lambda x: {
            'high': 0, 'medium': 1, 'low': 2
        }[x['severity']])
        
        self.explored_issues = {issue['type']: {'explored': False, 'root_causes': []} for issue in self.current_issues}
        self.current_issue_index = 0
        self.follow_up_count = 0
        self.root_causes = {}
        self.themes = set()
        self.potential_solutions = {}
        
        # Create issue context for the greeting
        issue_context = "\n".join([f"- {issue['type'].capitalize()}: {issue['description']}" 
                                 for issue in self.current_issues])
        
        # Generate supportive greeting
        greeting_prompt = f"""
        You are a supportive, solution-oriented HR chatbot having a conversation with an employee.
        
        The employee has these potential areas to discuss:
        {issue_context}
        
        Generate a warm, empathetic greeting that:
        1. Welcomes the employee and expresses genuine interest in helping them
        2. Mentions you're here to understand their challenges AND find solutions together
        3. Conveys a supportive, problem-solving attitude
        4. Notes they can type /exit anytime to end the conversation
        
        The greeting should be friendly, supportive and under 3 sentences.
        """
        
        greeting_prompt = sanitize_text(greeting_prompt)
        
        response = self.model.generate_content(
            contents=greeting_prompt,
            model="gemini-2.0-flash"
        )
        
        greeting = sanitize_text(response.text.strip() if response.text else 
                               "Hi there! I'm here to chat...")
        
        self.conversation.append({
            "role": "assistant",
            "content": greeting
        })
        
        return greeting

    def generate_question(self, conversation_history=None):
        """Generate a helpful, solution-oriented question that's concise"""
        if conversation_history is None:
            conversation_history = []
        
        # Convert conversation history to a readable format
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history[-3:]
        ]) if conversation_history else ""
        
        # Get current or unexplored issues
        current_issue = self.current_issues[self.current_issue_index] if self.current_issue_index < len(self.current_issues) else None
        unexplored_issues = [issue for issue in self.current_issues 
                           if not self.explored_issues[issue['type']]['explored']]
        
        # Create appropriate prompt based on conversation state
        if not history_text:
            # First question - start with high priority issue
            issue_type = current_issue['type']
            issue_description = current_issue['description']
            
            prompt = f"""
            You're a supportive, solution-oriented HR chatbot starting a conversation about {issue_type} ({issue_description}).
            
            Generate a brief, helpful question that:
            1. Is under 15 words but shows genuine interest in helping
            2. Feels natural and empathetic 
            3. Encourages them to share challenges so you can help find solutions
            4. Doesn't directly state that you identified an issue
            
            Write only the question itself, with no additional text.
            """
        else:
            # Follow-up or transition question
            prompt = f"""
            You're a supportive, solution-oriented HR chatbot continuing a conversation.
            
            RECENT CONVERSATION:
            {history_text}
            
            UNEXPLORED ISSUES: {[f"{i['type']}: {i['description']}" for i in unexplored_issues[:2]]}
            
            Generate a brief, helpful follow-up question that:
            1. Is under 15 words but shows genuine interest in helping solve their issues
            2. Naturally builds on their last response
            3. Explores potential solutions or ways you can help them
            4. Makes them feel supported and understood
            
            Write only the question itself, with no additional text.
            """
        
        prompt = sanitize_text(prompt)
        
        try:
            response = self.model.generate_content(
                contents=prompt,
                model="gemini-2.0-flash"
            )
            return sanitize_text(response.text.strip())
        except Exception as e:
            print(f"Error generating question: {e}")
            return "How can I help with this challenge you're facing?"

    def analyze_response(self, user_input, recent_conversation):
        """Analyze user response to identify root causes, themes, potential solutions"""
        # Convert recent conversation to text
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in recent_conversation
        ])
        
        # Format latest input for emphasis
        latest_input = f"LATEST USER INPUT:\n{sanitize_text(user_input)}\n\n"
        
        # Get current issue and all issues
        current_issue = self.current_issues[self.current_issue_index] if self.current_issue_index < len(self.current_issues) else None
        all_issues_text = "\n".join([
            f"- {issue['type']}: {issue['description']}" 
            for issue in self.current_issues
        ])
        
        # Analysis prompt
        analysis_prompt = f"""
        You are an HR analytics expert analyzing an employee's response about workplace issues.
        
        {latest_input}CURRENT CONVERSATION:
        {conversation_text}
        
        POTENTIAL ISSUES:
        {all_issues_text}
        
        CURRENTLY FOCUSED ISSUE: {current_issue['type'] if current_issue else 'None'} - {current_issue['description'] if current_issue else ''}
        
        Based on the employee's latest response, analyze:
        
        1. ROOT_CAUSES: What specific root causes did they mention for any issues? (Map issue types to lists of causes)
        2. THEMES: What broader themes emerged (work-life balance, communication, etc.)?
        3. POTENTIAL_SOLUTIONS: What specific solutions could help address the issues mentioned? (Map issue types to lists of solutions)
        4. SUFFICIENT_DEPTH: Has the current issue been explored sufficiently? (yes/no)
        5. SENTIMENT: What's the employee's sentiment about each issue mentioned? (Map issue types to sentiment)
        
        Format your response as JSON with these 5 keys.
        """
        
        analysis_prompt = sanitize_text(analysis_prompt)
        
        try:
            response = self.model.generate_content(
                contents=analysis_prompt,
                model="gemini-2.0-flash"
            )
            
            # Extract text, then sanitize, then process
            analysis_text = response.text
            analysis_text = sanitize_text(analysis_text)
            analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
            
            try:
                analysis = json.loads(analysis_text)
                
                # Update root causes
                if "ROOT_CAUSES" in analysis and isinstance(analysis["ROOT_CAUSES"], dict):
                    for issue_type, causes in analysis["ROOT_CAUSES"].items():
                        if issue_type not in self.root_causes:
                            self.root_causes[issue_type] = []
                        
                        if isinstance(causes, list):
                            for cause in causes:
                                if cause not in self.root_causes[issue_type]:
                                    self.root_causes[issue_type].append(cause)
                        elif isinstance(causes, str) and causes not in self.root_causes[issue_type]:
                            self.root_causes[issue_type].append(causes)
                
                # Update themes
                if "THEMES" in analysis and isinstance(analysis["THEMES"], list):
                    for theme in analysis["THEMES"]:
                        self.themes.add(theme)
                
                # Store potential solutions
                if "POTENTIAL_SOLUTIONS" in analysis and isinstance(analysis["POTENTIAL_SOLUTIONS"], dict):
                    for issue_type, solutions in analysis["POTENTIAL_SOLUTIONS"].items():
                        if issue_type not in self.potential_solutions:
                            self.potential_solutions[issue_type] = []
                        
                        if isinstance(solutions, list):
                            for solution in solutions:
                                if solution not in self.potential_solutions[issue_type]:
                                    self.potential_solutions[issue_type].append(solution)
                        elif isinstance(solutions, str) and solutions not in self.potential_solutions[issue_type]:
                            self.potential_solutions[issue_type].append(solutions)
                
                # Store sentiment data
                if "SENTIMENT" in analysis and isinstance(analysis["SENTIMENT"], dict):
                    for issue_type, sentiment in analysis["SENTIMENT"].items():
                        if issue_type not in self.sentiment_data:
                            self.sentiment_data[issue_type] = []
                        self.sentiment_data[issue_type].append(sentiment)
                
                # Check for sufficient depth with fallback
                sufficient_depth = "no"
                if "SUFFICIENT_DEPTH" in analysis:
                    sufficient_depth = analysis["SUFFICIENT_DEPTH"].lower() if isinstance(analysis["SUFFICIENT_DEPTH"], str) else "no"
                
                # Mark the current issue as explored if sufficient depth achieved
                if current_issue and sufficient_depth == "yes":
                    self.explored_issues[current_issue['type']]['explored'] = True
                    self.explored_issues[current_issue['type']]['root_causes'] = self.root_causes.get(current_issue['type'], [])
                
                return analysis
                    
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing error in analyze_response: {e}")
                return {
                    "ROOT_CAUSES": {}, "THEMES": [], "POTENTIAL_SOLUTIONS": {},
                    "SUFFICIENT_DEPTH": "no", "SENTIMENT": {}
                }
                    
        except Exception as e:
            print(f"Warning: Error analyzing response: {e}")
            return {
                "ROOT_CAUSES": {}, "THEMES": [], "POTENTIAL_SOLUTIONS": {},
                "SUFFICIENT_DEPTH": "no", "SENTIMENT": {}
            }

    def generate_solution_summary(self):
        """Generate a helpful summary of potential solutions for the identified issues"""
        # If no potential solutions were identified, generate basic recommendations
        if not self.potential_solutions and not self.root_causes:
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation])
            issues_text = "\n".join([f"- {issue['type']}: {issue['description']}" for issue in self.current_issues])
            
            solution_prompt = f"""
            You are an HR chatbot providing practical recommendations to an employee.
            
            Review this conversation about workplace issues:

            ISSUES:
            {issues_text}

            CONVERSATION:
            {conversation_text}

            Generate 2-3 practical, supportive recommendations that:
            1. Address the employee's main concerns
            2. Offer actionable steps they can take 
            3. Include ways the company can support them
            
            FORMAT: Brief, supportive bullet points that are specific and empathetic.
            Provide ONLY the final recommendations.
            """
            
            solution_prompt = sanitize_text(solution_prompt)
            
            try:
                response = self.model.generate_content(
                    contents=solution_prompt,
                    model="gemini-2.0-flash"
                )
                result = sanitize_text(response.text.strip())
                
                # Check if response contains any analysis markers and remove them
                if "option" in result.lower() or "options" in result.lower():
                    # Fallback to basic format
                    result = "Based on our conversation, here are some recommendations:\n\n• Schedule a meeting with your manager to discuss workload concerns\n• Consider documenting specific examples of challenges you're facing\n• Explore our company's well-being resources for additional support"
                
                return result
            except Exception:
                return "Based on our conversation, I recommend discussing your concerns with your manager and considering time management strategies that could help reduce stress."
        
        # Process identified solutions and root causes
        prompt = f"""
        You are an HR chatbot providing specific recommendations to an employee.

        Create supportive, practical recommendations based on:

        ROOT CAUSES:
        {json.dumps(self.root_causes, indent=2)}
        
        POTENTIAL SOLUTIONS:
        {json.dumps(self.potential_solutions, indent=2)}
                
        ISSUES:
        {[{'type': issue['type'], 'description': issue['description']} for issue in self.current_issues]}

        INSTRUCTIONS:
        1. Generate 3-4 specific, actionable recommendations
        2. Include both employee actions and company support options
        3. Format as brief, clear bullet points
        4. Be specific and practical with no fluff
        
        Provide ONLY the final recommendation text without analysis or explanation.
        """
        
        prompt = sanitize_text(prompt)
        
        try:
            response = self.model.generate_content(
                contents=prompt,
                model="gemini-2.0-flash"
            )
            result = sanitize_text(response.text.strip())
            
            # Verify the response doesn't contain option listings
            if "option" in result.lower() or "**option" in result.lower():
                # Fallback to basic solution format
                solution_text = "Based on our conversation, here are some recommendations:\n\n"
                for issue_type, solutions in self.potential_solutions.items():
                    if solutions:
                        issue_data = next((i for i in self.current_issues if i['type'] == issue_type), None)
                        if issue_data and solutions[0]:
                            solution_text += f"• For {issue_data['description'].lower()}: {solutions[0]}\n"
                return solution_text
            
            return result
        except Exception:
            # Fallback to basic solution summary
            solution_text = "Based on our conversation, here are some recommendations:\n\n"
            for issue_type, solutions in self.potential_solutions.items():
                if solutions:
                    issue_data = next((i for i in self.current_issues if i['type'] == issue_type), None)
                    if issue_data and solutions[0]:
                        solution_text += f"• For {issue_data['description'].lower()}: {solutions[0]}\n"
            return solution_text

    def run(self, issues):
        """Run an interactive conversation about the identified issues"""
        # Start conversation with greeting
        greeting = self.start_conversation(issues)
        print(f"Assistant: {greeting}")
        
        conversation_complete = False
        while not conversation_complete:
            # Get user input
            user_input = input("You: ")
            user_input = sanitize_text(user_input)
            
            # Check for exit command
            if user_input.strip().lower() == "/exit":
                farewell = "Thank you for your time. Let me share some suggestions based on our conversation before you go."
                solution_summary = self.generate_solution_summary()
                combined_message = f"{farewell}\n\n{solution_summary}"
                self.conversation.append({
                    "role": "assistant",
                    "content": combined_message
                })
                print(f"Assistant: {combined_message}")
                break
            
            # Add user input to conversation history
            self.conversation.append({
                "role": "user",
                "content": user_input
            })
            
            # Analyze the response
            analysis = self.analyze_response(user_input, self.conversation[-4:])
            
            # Move to next issue if current one is sufficiently explored
            if self.current_issue_index < len(self.current_issues):
                current_issue = self.current_issues[self.current_issue_index]
                sufficient_depth = analysis.get("SUFFICIENT_DEPTH", "no").lower() == "yes"
                
                if sufficient_depth or self.follow_up_count >= self.max_follow_ups:
                    # Mark current issue as explored
                    if current_issue:
                        self.explored_issues[current_issue['type']]['explored'] = True
                    # Move to next issue
                    self.current_issue_index += 1
                    self.follow_up_count = 0
                else:
                    # Continue with follow-up questions on current issue
                    self.follow_up_count += 1
            
            # Check if all issues have been explored
            all_explored = all(data['explored'] for data in self.explored_issues.values())
            
            # Generate next question or wrap up
            if all_explored or self.current_issue_index >= len(self.current_issues):
                # Generate solutions based on the conversation
                solutions = self.generate_solution_summary()
                
                # Create a helpful closing message with solutions
                closing_message = f"""Thank you for sharing your thoughts and experiences. Based on our conversation, I have some suggestions that might help:
                
{solutions}

I hope these recommendations are helpful. Your feedback is valuable and will help us create better workplace solutions."""
                
                self.conversation.append({
                    "role": "assistant",
                    "content": closing_message
                })
                print(f"Assistant: {sanitize_text(closing_message)}")

                conversation_complete = True
            else:
                # Generate next question based on updated conversation history
                next_question = self.generate_question(self.conversation[-6:])
                
                # Add to conversation
                self.conversation.append({
                    "role": "assistant",
                    "content": next_question
                })
                print(f"Assistant: {next_question}")
        
        return self.conversation

# 3. Report Generator Agent
class ReportGeneratorAgent:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        """Lazily initialize and return the model client with current API key"""
        if self._model is None:
            api_key = get_api_key()
            if not api_key:
                raise ValueError("No valid Gemini API key found for report generation")
            self._model = SafeGeminiClient(api_key=api_key)
        return self._model
    
    def assess_mental_health_criticality(self, conversation_text, issues, metrics):
        """Analyze conversation and data to assess mental health criticality"""
        criticality_prompt = f"""
        As a mental health specialist, analyze this employee conversation and data for any concerning 
        signs of serious mental health issues that require immediate attention.
        
        CONVERSATION:
        {conversation_text[:2000]}
        
        METRICS:
        {json.dumps(metrics, indent=2)}
        
        IDENTIFIED ISSUES:
        {json.dumps(issues, indent=2)}
        
        Carefully analyze for indicators of:
        1. Work-related trauma
        2. Clinical depression signs
        3. Burnout beyond normal work stress
        4. Self-harm ideation or references
        5. Suicidal ideation (direct or indirect)
        6. Severe anxiety or panic attacks
        7. Feelings of hopelessness related to work
        
        Return a JSON object with:
        1. "risk_level": One of ["none", "low", "medium", "high", "critical"]
        2. "indicators": List of specific concerning phrases or patterns detected
        3. "urgent_action_required": Boolean indicating if immediate intervention is needed
        4. "recommendations": List of appropriate mental health support recommendations
        
        If no concerning signs are present, indicate a "none" or "low" risk level.
        """
        
        criticality_prompt = sanitize_text(criticality_prompt)

        try:
            response = self.model.generate_content(
                contents=criticality_prompt,
                model="gemini-2.0-flash"
            )
            # Process response
            criticality_text = sanitize_text(response.text)
            criticality_text = criticality_text.replace("```json", "").replace("```", "").strip()
            
            try:
                return json.loads(criticality_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "risk_level": "unknown",
                    "indicators": ["Unable to assess - data processing error"],
                    "urgent_action_required": False,
                    "recommendations": ["Conduct manual assessment", "Follow standard mental health protocols"]
                }
        except Exception as e:
            print(f"Error during criticality assessment: {e}")
            return {
                "risk_level": "unknown",
                "indicators": ["Error during assessment"],
                "urgent_action_required": False,
                "recommendations": ["Conduct manual assessment"]
            }

    def extract_metrics(self, graph_data, employee_id):
        """Extract key metrics from the knowledge graph data"""
        metrics = {}
        
        # Extract vibe metrics
        vibe_node = graph_data.get(f"{employee_id}_vibe", {})
        if vibe_node:
            vibe_scores = vibe_node.get('scores', [])
            metrics['average_vibe'] = sum(vibe_scores) / len(vibe_scores) if vibe_scores else 0
            metrics['vibe_trend'] = vibe_node.get('trend', 'unknown')
        
        # Extract activity metrics
        activity_node = graph_data.get(f"{employee_id}_activity", {})
        if activity_node:
            metrics['avg_work_hours'] = activity_node.get('avg_work_hours', 0)
            metrics['avg_meetings'] = activity_node.get('avg_meetings', 0)
            metrics['avg_messages'] = activity_node.get('avg_messages', 0)
            metrics['avg_emails'] = activity_node.get('avg_emails', 0)
        
        # Extract performance metrics
        performance_node = graph_data.get(f"{employee_id}_performance", {})
        if performance_node:
            metrics['performance_rating'] = performance_node.get('rating', 0)
            metrics['promotion_consideration'] = performance_node.get('promotion', 'unknown')
        
        # Extract leave metrics
        leave_node = graph_data.get(f"{employee_id}_leave", {})
        if leave_node:
            metrics['leave_count'] = leave_node.get('leave_count', 0)
            metrics['leave_days_total'] = leave_node.get('leave_days_total', 0)
            metrics['leave_types'] = leave_node.get('leave_types', {})
        
        # Extract rewards metrics
        rewards_node = graph_data.get(f"{employee_id}_rewards", {})
        if rewards_node:
            metrics['reward_count'] = rewards_node.get('reward_count', 0)
            metrics['rewards_points'] = rewards_node.get('rewards_points', 0)
        
        # Make everything JSON serializable
        return ensure_json_serializable(metrics)

    def generate_report(self, employee_id, knowledge_graph, issues, conversation):
        """Generate a structured employee report in JSON format with categorized issues"""
        import tempfile
        
        # Convert the knowledge graph to a dictionary for easier analysis
        graph_data = {node: dict(knowledge_graph.nodes[node]) for node in knowledge_graph.nodes}
        
        # Extract key metrics from the graph
        metrics = self.extract_metrics(graph_data, employee_id)
        
        # Convert conversation to text format for analysis
        conversation_text = sanitize_text("\n".join([
            f"{'Employee' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in conversation
        ]))
        
        # Analyze issues to determine which categories they fall into
        category_analysis_prompt = f"""
        Analyze this employee data and conversation to determine which problem categories are most relevant.
        
        METRICS:
        {json.dumps(metrics, indent=2)}
        
        ISSUES:
        {json.dumps(issues, indent=2)}
        
        CONVERSATION:
        {conversation_text[:1500]}
        
        CATEGORIES TO CONSIDER:
        1. Absenteeism - Based on leave patterns and frequency
        2. Engagement - Based on vibe scores and communication activity
        3. Performance - Based on performance ratings and feedback
        4. Burnout - Based on work hours, meeting load, and emotional state
        5. Recognition - Based on rewards, awards, and promotion consideration
        6. Onboarding - Based on initial experience, training, and mentorship
        7. Work-Life Balance - Based on work hours and leave patterns
        8. Support - Based on mentoring, training completion, and manager feedback
        
        For each category, determine:
        1. Relevance score (0-10, where 10 is highly relevant)
        2. Key factors that make this category relevant or not
        3. Supporting evidence from the data
        
        Return a JSON object with each category as a key and an object containing score, factors, and evidence as values.
        Include only categories with a score of 5 or higher.
        """
        
        category_analysis_prompt = sanitize_text(category_analysis_prompt)
        
        try:
            category_response = self.model.generate_content(
                contents=category_analysis_prompt,
                model="gemini-2.0-flash"
            )
            # Process response
            category_text = sanitize_text(category_response.text)
            category_text = category_text.replace("```json", "").replace("```", "").strip()
            category_analysis = json.loads(category_text)
        except Exception as e:
            print(f"Error analyzing categories: {e}")
            category_analysis = {}
        
        # Assess mental health criticality
        criticality_assessment = self.assess_mental_health_criticality(
            conversation_text, 
            issues,
            metrics
        )
        
        # Generate full report prompt with criticality included
        report_prompt = f"""
        Generate a professional HR report for employee {employee_id} based on the following data.
        
        METRICS:
        {json.dumps(metrics, indent=2)}
        
        IDENTIFIED ISSUES:
        {json.dumps(issues, indent=2)}
        
        PROBLEM CATEGORIES:
        {json.dumps(category_analysis, indent=2)}
        
        MENTAL HEALTH CRITICALITY:
        {json.dumps(criticality_assessment, indent=2)}
        
        CONVERSATION INSIGHTS:
        {conversation_text[:1500]}
        
        Generate a JSON structured report with these sections:
        1. executive_summary: Brief overview (2-3 sentences)
        2. key_metrics: Object with the most important metrics and their significance
        3. identified_issues: Array of issue objects with type, severity, and descriptions
        4. problem_categories: The analyzed categories with their scores and factors
        5. root_causes: Array of identified root causes grouped by issue type
        6. recommended_actions: Array of specific, actionable recommendations
        7. criticality: The mental health criticality assessment object
        
        Return ONLY valid JSON with no additional text or explanation.
        """
        
        report_prompt = sanitize_text(report_prompt)
        
        try:
            response = self.model.generate_content(
                contents=report_prompt,
                model="gemini-2.0-flash"
            )
            # Process response
            report_text = sanitize_text(response.text)
            report_text = report_text.replace("```json", "").replace("```", "").strip()
            
            try:
                json_report = json.loads(report_text)
                # Ensure criticality is included
                if "criticality" not in json_report:
                    json_report["criticality"] = criticality_assessment
                
                # Ensure the report is fully serializable
                serializable_report = ensure_json_serializable(json_report)
                
                # Create JSON file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_json:
                    json.dump(serializable_report, f_json, indent=2)
                    report_json_path = f_json.name
                
                # Create a readable text summary
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f_txt:
                    f_txt.write(f"EMPLOYEE REPORT: {employee_id}\n\n")
                    f_txt.write(f"EXECUTIVE SUMMARY:\n{json_report.get('executive_summary', 'No summary available')}\n\n")
                    
                    # Write criticality section
                    f_txt.write("MENTAL HEALTH CRITICALITY:\n")
                    criticality = json_report.get('criticality', {})
                    f_txt.write(f"- Risk Level: {criticality.get('risk_level', 'Unknown')}\n")
                    f_txt.write(f"- Urgent Action Required: {criticality.get('urgent_action_required', False)}\n")
                    f_txt.write("- Indicators:\n")
                    for indicator in criticality.get('indicators', []):
                        f_txt.write(f"  • {indicator}\n")
                    f_txt.write("- Recommendations:\n")
                    for rec in criticality.get('recommendations', []):
                        f_txt.write(f"  • {rec}\n")
                    f_txt.write("\n")
                    
                    # Add remaining sections
                    f_txt.write("KEY METRICS:\n")
                    for metric, value in json_report.get('key_metrics', {}).items():
                        f_txt.write(f"- {metric}: {value}\n")
                    f_txt.write("\n")
                    
                    f_txt.write("IDENTIFIED ISSUES:\n")
                    for issue in json_report.get('identified_issues', []):
                        f_txt.write(f"- {issue.get('type', 'Unknown')} ({issue.get('severity', 'medium')}): {issue.get('description', 'No description')}\n")
                    f_txt.write("\n")
                    
                    f_txt.write("PROBLEM CATEGORIES:\n")
                    for category, details in json_report.get('problem_categories', {}).items():
                        f_txt.write(f"- {category} (Score: {details.get('score', 'N/A')})\n")
                        f_txt.write(f"  Factors: {', '.join(details.get('factors', []))}\n")
                    f_txt.write("\n")
                    
                    f_txt.write("ROOT CAUSES:\n")
                    for cause in json_report.get('root_causes', []):
                        f_txt.write(f"- {cause}\n")
                    f_txt.write("\n")
                    
                    f_txt.write("RECOMMENDED ACTIONS:\n")
                    for action in json_report.get('recommended_actions', []):
                        f_txt.write(f"- {action}\n")
                    
                    report_txt_path = f_txt.name
                
                print(f"Report saved as JSON: {report_json_path}")
                print(f"Report saved as readable text: {report_txt_path}")
                
                # Return the report object
                return serializable_report
                
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON generated: {e}")
                # Create a basic JSON structure
                json_report = {
                    "executive_summary": "Error generating structured report.",
                    "key_metrics": metrics,
                    "identified_issues": issues,
                    "problem_categories": category_analysis,
                    "root_causes": [],
                    "recommended_actions": ["Review data manually", "Schedule follow-up with employee"],
                    "criticality": criticality_assessment
                }
                return json_report
                
        except Exception as e:
            print(f"Error generating report content: {e}")
            return {
                "error": f"Error generating report for employee {employee_id}: {str(e)}",
                "key_metrics": metrics,
                "identified_issues": issues,
                "criticality": criticality_assessment
            }
    
    def run(self, employee_id, knowledge_graph, issues, conversation_history):
        """Run the report generator"""
        report = self.generate_report(employee_id, knowledge_graph, issues, conversation_history)
        print(f"Report generated successfully for employee {employee_id}")
        return report

# Define the LangGraph workflow
def build_workflow(activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df):
    # Initialize agents
    graph_builder = GraphBuilderAgent(
        activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df
    )
    chatbot = ChatbotAgent()
    report_generator = ReportGeneratorAgent()
    
    # Define node functions
    def build_graph(state):
        employee_id = state.get('employee_id')
        knowledge_graph, issues = graph_builder.run(employee_id)
        return {
            'employee_id': employee_id,
            'knowledge_graph': knowledge_graph,
            'issues': issues
        }
    
    def conduct_conversation(state):
        issues = state.get('issues', [])
        conversation_history = chatbot.run(issues)
        return {'conversation_history': conversation_history}
    
    def generate_report(state):
        employee_id = state.get('employee_id')
        knowledge_graph = state.get('knowledge_graph')
        issues = state.get('issues', [])
        conversation_history = state.get('conversation_history', [])
        report = report_generator.run(employee_id, knowledge_graph, issues, conversation_history)
        return {'report': report}
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes and edges
    workflow.add_node("build_graph", build_graph)
    workflow.add_node("conduct_conversation", conduct_conversation)
    workflow.add_node("generate_report", generate_report)
    
    workflow.add_edge("build_graph", "conduct_conversation")
    workflow.add_edge("conduct_conversation", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # Set the entry point
    workflow.set_entry_point("build_graph")
    
    return workflow.compile()

# Function to run the entire workflow
def run_hr_analysis(employee_id):
    # Load datasets
    activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df = load_datasets()
    
    # Build the workflow
    workflow = build_workflow(
        activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df
    )
    
    # Run the workflow
    return workflow.invoke({"employee_id": employee_id})

# Example usage
def main():
    try:
        # Load datasets
        activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df = load_datasets()
        
        # Default employee ID
        employee_id = 'EMP0387'
        
        # Create the agents directly
        graph_builder = GraphBuilderAgent(
            activity_df, leave_df, onboarding_df, performance_df, rewards_df, vibemeter_df
        )
        chatbot = ChatbotAgent()
        report_generator = ReportGeneratorAgent()
        
        # Build knowledge graph and identify issues
        knowledge_graph, issues = graph_builder.run(employee_id)
        print(f"Found {len(issues)} potential issues for Employee {employee_id}")
        
        # Run interactive conversation
        print("Starting interactive conversation...\n")
        conversation = chatbot.run(issues)
        
        # Generate report
        print("\nConversation complete! Generating report...")
        report = report_generator.run(employee_id, knowledge_graph, issues, conversation)
        print(f"Successfully generated report for employee {employee_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

