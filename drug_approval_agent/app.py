import sys
import os
# Add parent directory to path so Python can find the drug_approval_agent package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import plotly.express as px
import pandas as pd
from drug_approval_agent.api.drug_api import DrugApprovalAgent
from drug_approval_agent.utils.config import load_config
from drug_approval_agent.utils.helpers import format_date
from dotenv import load_dotenv

# First try to load from .env, then fall back to .env.template
env_path = os.path.join(os.path.dirname(__file__), '.env')
template_path = os.path.join(os.path.dirname(__file__), '.env.template')

if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
elif os.path.exists(template_path):
    load_dotenv(dotenv_path=template_path)
    st.warning("Using .env.template file. For production use, please create a .env file with your API keys.")
else:
    st.error("No .env or .env.template file found. API functionality may be limited.")

# Remove debug prints with API keys for security

# Define helper functions
def display_search_results(results):
    if not results:
        st.warning("No results found for your search criteria.")
        return
    
    # Filter out mock data
    real_results = [r for r in results if not r.get('is_mock_data', False)]
    
    # If no real results, show warning
    if not real_results:
        st.warning("No official drug data found. Try a different search term or date range.")
        return
    
    # Check if results are from web search
    web_results = [r for r in real_results if r.get('source') == 'Web Search']
    if web_results:
        st.info("Showing results from web search. These may not be as accurate as official FDA or clinical trials data.")
    
    # Display results as cards
    st.subheader(f"Found {len(real_results)} results")
    
    # Group by drug name
    grouped_results = {}
    for result in real_results:
        drug_name = result.get('drug_name', 'Unknown')
        if drug_name not in grouped_results:
            grouped_results[drug_name] = []
        grouped_results[drug_name].append(result)
    
    # Sort drugs by latest update date
    def get_latest_date(results_list):
        dates = [result.get('date') for result in results_list if result.get('date')]
        return max(dates) if dates else ""
    
    sorted_drugs = sorted(grouped_results.items(), key=lambda x: get_latest_date(x[1]), reverse=True)
    
    # Display cards for each drug
    for drug_name, drug_results in sorted_drugs:
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            
            # Get the most recent/relevant result for this drug
            primary_result = sorted(drug_results, key=lambda x: x.get('date', ''), reverse=True)[0]
            
            # Display drug header with manufacturer
            manufacturer = primary_result.get('manufacturer', 'Unknown Manufacturer')
            st.markdown(f"### {drug_name}")
            st.markdown(f"**Manufacturer:** {manufacturer}")
            
            # Display drug status and phase
            status = primary_result.get('status', 'Unknown Status')
            phase = primary_result.get('phase', '')
            st.markdown(f"**Status:** {status} {f'(Phase {phase})' if phase else ''}")
            
            # Display therapeutic area
            therapeutic_area = primary_result.get('therapeutic_area', 'Unknown')
            st.markdown(f"**Therapeutic Area:** {therapeutic_area}")
            
            # Display latest update date
            latest_date = get_latest_date(drug_results)
            if latest_date:
                st.markdown(f"**Latest Update:** {latest_date}")
            
            # Display description if available (from web search)
            if primary_result.get('description'):
                st.markdown(f"**Description:** {primary_result.get('description')}")
            
            # Display source information and links
            source_type = primary_result.get('type', 'Unknown')
            
            if source_type == 'FDA Approval':
                st.markdown(f"**Source:** FDA Database")
                application_number = primary_result.get('application_number', '')
                if application_number and application_number != 'Unknown':
                    st.markdown(f"[View FDA Record](https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={application_number})")
            
            elif source_type == 'Clinical Trial':
                st.markdown(f"**Source:** ClinicalTrials.gov")
                nct_id = primary_result.get('nct_id', '')
                if nct_id and nct_id != 'Unknown':
                    st.markdown(f"[View Clinical Trial](https://clinicaltrials.gov/study/{nct_id})")
            
            elif source_type == 'Web Search Result':
                st.markdown(f"**Source:** {primary_result.get('source', 'Web Search')}")
                url = primary_result.get('url', '')
                if url:
                    st.markdown(f"[View Source]({url})")
            
            # Display news mentions if available
            if primary_result.get('news_mentions'):
                st.markdown("#### Recent News")
                for mention in primary_result['news_mentions'][:2]:  # Show top 2 mentions
                    st.markdown(f"🔹 **{mention.get('title', '')}** - *{mention.get('source', '')}*")
                    if mention.get('url'):
                        st.markdown(f"[Read more]({mention.get('url')})")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="Drug Approval Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark mode
st.markdown("""
<style>
    :root {
        --background-color: #121212;
        --secondary-background-color: #1e1e1e;
        --text-color: #f0f0f0;
        --font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font-family);
    }
    
    .stSidebar {
        background-color: var(--secondary-background-color);
    }
    
    .stButton button {
        background-color: #1e88e5;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .card {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    h1, h2, h3 {
        color: #1e88e5;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1e88e5;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #b0b0b0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the agent
config = load_config()
agent = DrugApprovalAgent(config)

# App title
st.title("Drug Approval Intelligence")
st.markdown("### Explore drug approvals and pipelines with AI-powered insights")

# Sidebar
with st.sidebar:
    st.header("Search Options")
    
    search_query = st.text_input("Enter research topic", placeholder="e.g., cancer drugs, Pfizer pipeline, Alzheimer's treatments")
    
    st.subheader("Date Range")
    start_date = st.date_input("From", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("To", pd.to_datetime("now"))
    
    search_button = st.button("Search")
    
    st.markdown("---")
    st.markdown("### Ask AI Assistant")
    user_question = st.text_area("Ask a question about drug approvals", height=100)
    ask_button = st.button("Ask")

# Main content
if search_button and search_query:
    st.subheader(f"Results for: {search_query}")
    
    try:
        with st.spinner("Searching for drug information..."):
            # Search across all types
            all_results = []
            errors = []
            
            # Try each search type and combine results
            for search_type in ["Drug Name", "Manufacturer", "Therapeutic Area", "Approval Status"]:
                try:
                    type_results = agent.search_drugs(
                        query=search_query,
                        search_type=search_type,
                        start_date=start_date,
                        end_date=end_date
                    )
                    all_results.extend(type_results)
                except Exception as e:
                    errors.append(f"Error searching {search_type}: {str(e)}")
                    st.error(f"Error searching {search_type}: {str(e)}")
            
            # Remove duplicates by drug name
            unique_results = []
            seen_drugs = set()
            
            for result in all_results:
                drug_id = f"{result.get('drug_name', 'Unknown')}-{result.get('manufacturer', 'Unknown')}"
                if drug_id not in seen_drugs:
                    seen_drugs.add(drug_id)
                    unique_results.append(result)
            
            results = unique_results
        
        # Filter out mock data
        real_results = [r for r in results if not r.get('is_mock_data', False)]
        
        # If no results from APIs, try web search
        if not real_results:
            st.warning("No results found from official sources. Trying web search...")
            
            with st.spinner("Searching the web for drug information..."):
                web_results = []
                # Create a direct web search using the original query
                import requests
                
                # Create web search query
                web_query = search_query
                if "drug" not in web_query.lower() and "medicine" not in web_query.lower():
                    web_query += " drug"
                
                if "approval" not in web_query.lower() and "fda" not in web_query.lower():
                    web_query += " approval FDA"
                
                try:
                    # Use DuckDuckGo API for web search
                    response = requests.get(
                        "https://api.duckduckgo.com",
                        params={
                            "q": web_query,
                            "format": "json"
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Check if we got useful results
                        if data.get("AbstractText") or data.get("RelatedTopics"):
                            web_results = agent._fallback_to_web_search(
                                [search_query], 
                                format_date(start_date), 
                                format_date(end_date), 
                                10
                            )
                except Exception as e:
                    st.error(f"Web search error: {str(e)}")
                
                if web_results:
                    st.info("Found results from web search. These may not be as accurate as official FDA data.")
                    real_results = web_results
        
        # Update results with real data only
        results = real_results
        
        # Show web search info if applicable
        if any(result.get('source', '') == 'Web Search' for result in results):
            st.info("Some results are from web search rather than official FDA or clinical trials databases.")
    except Exception as e:
        st.error(f"An error occurred during the search: {str(e)}")
        results = []
    
    if results:
        # Convert results to DataFrame for visualization
        df = pd.DataFrame(results)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{len(df)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Total Results</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            approval_count = len(df[df['status'] == 'Approved'])
            st.markdown(f'<div class="metric-value">{approval_count}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Approved Drugs</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            pipeline_count = len(df[df['status'] != 'Approved'])
            st.markdown(f'<div class="metric-value">{pipeline_count}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Pipeline Drugs</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Cards", "Overview", "Pipeline Status", "News & Mentions"])
        
        with tab1:
            display_search_results(results)
            
        with tab2:
            if 'therapeutic_area' in df.columns:
                fig1 = px.pie(df, names='therapeutic_area', title='Distribution by Therapeutic Area')
                fig1.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            if 'manufacturer' in df.columns:
                top_manufacturers = df['manufacturer'].value_counts().reset_index()
                top_manufacturers.columns = ['Manufacturer', 'Count']
                fig2 = px.bar(
                    top_manufacturers.head(10), 
                    x='Count', 
                    y='Manufacturer',
                    title='Top Manufacturers',
                    orientation='h'
                )
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            if 'phase' in df.columns:
                phase_counts = df['phase'].value_counts().reset_index()
                phase_counts.columns = ['Phase', 'Count']
                phase_order = ['Preclinical', 'Phase 1', 'Phase 2', 'Phase 3', 'NDA/BLA Filed', 'Approved']
                phase_counts['Phase'] = pd.Categorical(phase_counts['Phase'], categories=phase_order, ordered=True)
                phase_counts = phase_counts.sort_values('Phase')
                
                fig3 = px.bar(
                    phase_counts, 
                    x='Phase', 
                    y='Count',
                    title='Drug Pipeline Status',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig3.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            # Timeline of approvals/updates
            if 'date' in df.columns:
                try:
                    # Convert date column to datetime type
                    timeline_df = df.copy()
                    timeline_df['date'] = pd.to_datetime(timeline_df['date'], errors='coerce')
                    
                    # Drop rows with invalid dates
                    timeline_df = timeline_df.dropna(subset=['date'])
                    
                    if not timeline_df.empty:
                        # Set date as index for proper grouping
                        timeline_df = timeline_df.set_index('date')
                        # Group by month
                        monthly_counts = timeline_df.groupby(pd.Grouper(freq='M')).size().reset_index(name='count')
                        
                        fig4 = px.line(
                            monthly_counts,
                            x='date',
                            y='count',
                            title='Timeline of Drug Updates'
                        )
                        fig4.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.info("Not enough timeline data to display chart")
                except Exception as e:
                    st.error(f"Error generating timeline: {str(e)}")
                    st.info("Could not generate timeline visualization due to date format issues")
        
        with tab4:
            if 'news_mentions' in df.columns:
                st.subheader("Recent News & Mentions")
                for idx, row in df.iterrows():
                    if row.get('news_mentions'):
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"#### {row['drug_name']} ({row['manufacturer']})")
                        for mention in row['news_mentions'][:3]:  # Show top 3 mentions
                            st.markdown(f"🔹 **{mention['title']}**")
                            st.markdown(f"*{mention['source']} - {mention['date']}*")
                            st.markdown(f"{mention['summary']}")
                            st.markdown(f"[Read more]({mention['url']})")
                        st.markdown("</div>", unsafe_allow_html=True)
        
        # Detailed data table
        st.subheader("Detailed Results")
        # Determine which columns to display based on source
        if any(r.get('source') == 'Web Search' for r in results):
            # For web search results, include the URL
            display_columns = ['drug_name', 'manufacturer', 'therapeutic_area', 'status', 'phase', 'source', 'url']
        else:
            # For API results
            display_columns = ['drug_name', 'manufacturer', 'therapeutic_area', 'status', 'phase', 'date']

        # Create a DataFrame with only the columns that exist in the results
        display_df = pd.DataFrame(results)
        available_columns = [col for col in display_columns if col in display_df.columns]
        display_df = display_df[available_columns]

        # Make URLs clickable if present
        if 'url' in display_df.columns:
            # Create a temporary column with markdown links
            display_df['URL'] = display_df['url'].apply(lambda x: f"[Link]({x})" if pd.notna(x) and x else "")
            # Replace url column with the markdown version
            display_df = display_df.drop('url', axis=1)
            display_df = display_df.rename(columns={'URL': 'url'})

        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.info("No results found. Try adjusting your search criteria.")

# AI Assistant section
if ask_button and user_question:
    st.subheader("AI Assistant Response")
    
    with st.spinner("Analyzing your question..."):
        answer = agent.answer_question(user_question)
    
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    st.markdown(answer["answer"])
    
    # Show sources if available
    if answer.get("sources"):
        st.markdown("**Sources:**")
        for source in answer["sources"]:
            st.markdown(f"- [{source['title']}]({source['url']})")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Show a welcome message if no action has been taken
if not (search_button and search_query) and not (ask_button and user_question):
    st.info("👋 Welcome! Search for drug information or ask the AI assistant about drug approvals and pipelines.")
    
    # Sample questions
    st.markdown("### Sample questions to ask:")
    st.markdown("- What are the latest FDA approved drugs for cancer?")
    st.markdown("- What's in Pfizer's drug pipeline for 2023?")
    st.markdown("- Which drugs have been approved for Alzheimer's in the last 2 years?")
    st.markdown("- What is the status of CAR-T therapies in clinical trials?")
    st.markdown("- Which companies are leading in rare disease drug development?")

if __name__ == "__main__":
    # This section is executed when the script is run directly
    pass 