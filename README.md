# 🚀 HR Analytics Assistant

> Your AI-powered HR sidekick for employee engagement and wellbeing insights

HR Analytics Assistant is an elegant, intelligent tool that transforms raw employee data into meaningful insights. Using advanced AI, it identifies potential issues, facilitates supportive conversations, and generates actionable recommendations to enhance workplace wellbeing.

This project has also been hosted on Huggingface Spaces, you can check it out: https://huggingface.co/spaces/drcocktail/HR-AGENT-DEV

![HR Analytics Assistant Demo](https://github.com/yourusername/hr-analytics-assistant/assets/demo.gif)

## ✨ Features

- **Employee Knowledge Graph** - Build comprehensive employee profiles by connecting data from multiple sources
- **Issue Detection** - Automatically identify potential issues like declining vibe scores, excessive workload, or recognition gaps
- **AI-Powered Chat** - Conduct empathetic, solution-focused conversations with employees
- **Interactive Analytics** - Visualize employee metrics through beautiful, information-rich dashboards
- **Custom Report Builder** - Generate tailored reports with selected components based on your needs
- **Mental Health Assessment** - Identify potential wellbeing concerns with appropriate recommendations

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Visualization**: Matplotlib, Altair
- **AI Integration**: Google Gemini AI
- **Data Processing**: Pandas, NetworkX
- **Architecture**: LangGraph for agent orchestration

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hr-analytics-assistant.git
   cd hr-analytics-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key:
   ```bash
   # Create a .env file with your Gemini API key
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

## 🚀 Usage

1. Start the Streamlit app:
   ```bash
   streamlit run src/streamlit_app.py
   ```

2. Choose data source:
   - Use the included sample data
   - Upload your own CSV files

3. Select an employee ID to analyze

4. Navigate through the tabs:
   - **Chat**: Have an AI-powered supportive conversation
   - **Analytics Dashboard**: Explore visualizations of key metrics
   - **Report Builder**: Create customized reports
   - **Full Report**: View comprehensive analysis and recommendations

## 📊 Data Requirements

The application expects the following datasets:

- **Activity Tracker**: Work hours, meetings, communication metrics
- **Leave Data**: Time off, leave types, patterns
- **Onboarding Information**: Joining date, training completion, mentorship
- **Performance Metrics**: Ratings, feedback, promotion consideration
- **Rewards Data**: Recognition, points, awards
- **Vibemeter**: Employee satisfaction scores over time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

- Integration with HRIS systems for real-time data
- Enhanced conversation abilities with memory across sessions
- PDF export functionality for reports
- Team-level analytics for organizational insights
- Slack/MS Teams integration for seamless workplace adoption

---

*Built with ❤️ for HR professionals who care about employee wellbeing* 
