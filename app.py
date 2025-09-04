"""
HireWithAI - Smart Resume Screening System
A CrewAI-powered multi-agent recruitment platform using GROQ API for fast inference.

This single-file application includes three AI agents:
1. Resume Parser Agent - Extracts structured data from resumes
2. Skill Matcher Agent - Matches skills to job descriptions  
3. Ranking Agent - Ranks candidates based on relevance

Author: AI Developer
License: MIT
"""

import streamlit as st
import tempfile
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import warnings
import time
import asyncio
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

# Core imports for AI agents
try:
    from crewai import Agent, Task, Crew, LLM
    from crewai.project import CrewBase, agent, crew, task
    from groq import Groq
    import spacy
    from spacy.matcher import PhraseMatcher
    import PyPDF2
    import docx2txt
    import re
    import hashlib
except ImportError as e:
    st.error(f"Missing required dependency: {e}")
    st.stop()

# Configuration - Single model only
GROQ_MODEL = "llama-3.1-8b-instant"
MODEL_DISPLAY_NAME = "Llama 3.1 8B (Fastest)"

# Rate limiting configuration
RATE_LIMIT_DELAY = 3  # seconds between requests
MAX_RETRIES = 3
BATCH_SIZE = 2  # Process resumes in smaller batches

# Initialize session state
if 'processed_resumes' not in st.session_state:
    st.session_state.processed_resumes = []
if 'ranked_candidates' not in st.session_state:
    st.session_state.ranked_candidates = []
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""

class RateLimitHandler:
    """Handle rate limiting for API calls"""
    
    def __init__(self, delay=RATE_LIMIT_DELAY):
        self.delay = delay
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.delay:
            sleep_time = self.delay - time_since_last_request
            st.info(f"⏳ Rate limit protection: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

class ResumeProcessor:
    """Utility class for processing resume files"""
    
    @staticmethod
    def extract_text_from_pdf(file_buffer) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file_buffer)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_buffer) -> str:
        """Extract text from DOCX file"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_buffer.read())
                tmp_file.flush()
                text = docx2txt.process(tmp_file.name)
                os.unlink(tmp_file.name)
                return text.strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_buffer) -> str:
        """Extract text from TXT file"""
        try:
            return file_buffer.read().decode('utf-8').strip()
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""

class HireWithAICrew:
    """Main CrewAI multi-agent system for resume screening with rate limiting"""
    
    def __init__(self, groq_api_key: str):
        """Initialize the crew with GROQ API and rate limiting"""
        self.llm = LLM(
            model=f"groq/{GROQ_MODEL}",
            api_key=groq_api_key,
            temperature=0.1
        )
        
        self.rate_limiter = RateLimitHandler()
        
        # Initialize spaCy for NLP operations
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")
            st.stop()
    
    def create_resume_parser_agent(self) -> Agent:
        """Create the Resume Parser Agent"""
        return Agent(
            role='Resume Parser Specialist',
            goal='Extract key candidate information from resume text efficiently',
            backstory="""You are an expert resume parser focused on extracting essential 
            candidate information quickly and accurately. You prioritize the most important 
            details: name, contact info, skills, experience, and education.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_skill_matcher_agent(self) -> Agent:
        """Create the Skill Matcher Agent"""
        return Agent(
            role='Skill Matching Expert',
            goal='Efficiently match candidate skills with job requirements',
            backstory="""You are a skilled matcher who quickly identifies relevant skills 
            and calculates match percentages. You focus on the most critical skills 
            and provide concise, actionable insights.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_ranking_agent(self) -> Agent:
        """Create the Ranking Agent"""
        return Agent(
            role='Candidate Ranking Analyst',
            goal='Rank candidates efficiently based on key criteria',
            backstory="""You are a recruitment analyst who creates fast, accurate candidate 
            rankings. You focus on the most important factors: skills match, experience 
            relevance, and overall job fit.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_concise_parsing_task(self, resume_text: str, filename: str) -> Task:
        """Create a more concise parsing task to reduce token usage"""
        # Truncate resume text if too long to save tokens
        max_chars = 2000
        truncated_text = resume_text[:max_chars] + "..." if len(resume_text) > max_chars else resume_text
        
        return Task(
            description=f"""
            Extract key information from this resume in JSON format:
            
            Resume: {filename}
            Text: {truncated_text}
            
            Extract:
            1. Name and contact (email, phone)
            2. Key skills (top 5-8 most relevant)
            3. Experience summary (years, key roles)
            4. Education (degree, field)
            5. Notable achievements
            
            Keep response concise and structured.
            """,
            expected_output="Concise JSON with essential candidate information",
            agent=self.create_resume_parser_agent()
        )
    
    def create_concise_skill_matching_task(self, resume_data: str, job_description: str) -> Task:
        """Create a more concise skill matching task"""
        # Truncate job description if too long
        max_jd_chars = 1000
        truncated_jd = job_description[:max_jd_chars] + "..." if len(job_description) > max_jd_chars else job_description
        
        return Task(
            description=f"""
            Analyze skill match between candidate and job:
            
            Job Requirements: {truncated_jd}
            Candidate Data: {resume_data}
            
            Provide:
            1. Match percentage (0-100%)
            2. Top 5 matching skills
            3. Top 3 missing critical skills
            4. Experience level fit (1-10)
            
            Keep analysis concise and focused.
            """,
            expected_output="Concise skill matching analysis in JSON format",
            agent=self.create_skill_matcher_agent()
        )
    
    def safe_crew_execution(self, crew, max_retries=MAX_RETRIES):
        """Execute crew with retry logic for rate limits"""
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                result = crew.kickoff()
                return result
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "ratelimit" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Progressive backoff
                        st.warning(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        st.error("Maximum retries reached. Please try again later or upgrade your Groq plan.")
                        return None
                else:
                    st.error(f"Error during processing: {e}")
                    return None
        return None
    
    def process_resumes_with_batching(self, resumes_data: List[Dict], job_description: str) -> Dict:
        """Process resumes in smaller batches to avoid rate limits"""
        try:
            all_parsed_resumes = []
            all_skill_analysis = []
            
            # Process resumes in batches
            total_resumes = len(resumes_data)
            batches = [resumes_data[i:i + BATCH_SIZE] for i in range(0, total_resumes, BATCH_SIZE)]
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            for batch_idx, batch in enumerate(batches):
                progress_text.text(f"Processing batch {batch_idx + 1} of {len(batches)}...")
                
                # Step 1: Parse resumes in current batch
                for resume_idx, resume_data in enumerate(batch):
                    overall_progress = (batch_idx * BATCH_SIZE + resume_idx) / total_resumes
                    progress_bar.progress(overall_progress)
                    
                    parsing_task = self.create_concise_parsing_task(
                        resume_data['text'], 
                        resume_data['filename']
                    )
                    
                    parsing_crew = Crew(
                        agents=[self.create_resume_parser_agent()],
                        tasks=[parsing_task],
                        verbose=False  # Reduce verbosity to save tokens
                    )
                    
                    result = self.safe_crew_execution(parsing_crew)
                    if result:
                        all_parsed_resumes.append({
                            'filename': resume_data['filename'],
                            'parsed_data': result.raw,
                            'original_text': resume_data['text'][:500]  # Store only first 500 chars
                        })
                
                # Step 2: Skill matching for current batch
                for resume in all_parsed_resumes[-len(batch):]:  # Only process newly added resumes
                    skill_task = self.create_concise_skill_matching_task(
                        resume['parsed_data'],
                        job_description
                    )
                    
                    skill_crew = Crew(
                        agents=[self.create_skill_matcher_agent()],
                        tasks=[skill_task],
                        verbose=False
                    )
                    
                    result = self.safe_crew_execution(skill_crew)
                    if result:
                        all_skill_analysis.append({
                            'filename': resume['filename'],
                            'skill_analysis': result.raw,
                            'parsed_data': resume['parsed_data']
                        })
            
            progress_bar.progress(1.0)
            progress_text.text("Finalizing rankings...")
            
            # Step 3: Final ranking (only if we have successful analyses)
            if all_skill_analysis:
                # Create a more concise ranking task
                ranking_task = Task(
                    description=f"""
                    Rank these candidates for the job. Provide top 5 ranked candidates with scores.
                    
                    Job: {job_description[:500]}...
                    
                    Candidates: {json.dumps([sa['skill_analysis'] for sa in all_skill_analysis[:5]], indent=1)}
                    
                    Provide concise ranking with:
                    1. Candidate name and rank
                    2. Overall score (0-100)
                    3. Key strengths (2-3 points)
                    4. Brief recommendation
                    """,
                    expected_output="Concise candidate ranking with top recommendations",
                    agent=self.create_ranking_agent()
                )
                
                ranking_crew = Crew(
                    agents=[self.create_ranking_agent()],
                    tasks=[ranking_task],
                    verbose=False
                )
                
                ranking_result = self.safe_crew_execution(ranking_crew)
                final_ranking = ranking_result.raw if ranking_result else "Ranking failed due to rate limits"
            else:
                final_ranking = "No candidates could be analyzed due to rate limits"
            
            return {
                'parsed_resumes': all_parsed_resumes,
                'skill_analysis': all_skill_analysis,
                'final_ranking': final_ranking
            }
            
        except Exception as e:
            st.error(f"Error processing resumes: {e}")
            return {}

def main():
    """Main Streamlit application"""
    
    # Page config
    st.set_page_config(
        page_title="HireWithAI - Smart Resume Screening",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .rate-limit-info {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 HireWithAI - Smart Resume Screening System</h1>
        <p>AI-Powered Multi-Agent Recruitment Platform</p>
        <p><i>Reduce 70% of time-to-hire with automated resume screening and ranking</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # GROQ API Key
        groq_api_key = st.text_input(
            "GROQ API Key",
            type="password",
            help="Get your free API key from https://console.groq.com/"
        )
        
        if not groq_api_key:
            st.warning("Please enter your GROQ API key to continue")
            st.info("💡 **Get Free GROQ API Key:**\n1. Visit https://console.groq.com/\n2. Create an account\n3. Generate API key\n4. Paste it above")
            return
        
        # Model info (fixed model)
        st.markdown("### 🤖 AI Model")
        st.info(f"**Using:** {MODEL_DISPLAY_NAME}")
        st.caption("Optimized for speed and efficiency")
        
        # Rate limiting info
        st.markdown("### ⚡ Rate Limiting")
        st.markdown("""
        <div class="rate-limit-info">
            <strong>🛡️ Built-in Protection:</strong><br>
            • Smart batch processing<br>
            • Automatic retry logic<br>
            • Progressive delays<br>
            • Token usage optimization
        </div>
        """, unsafe_allow_html=True)
        
        # Processing statistics
        st.header("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Resumes Processed", len(st.session_state.processed_resumes))
        with col2:
            st.metric("Candidates Ranked", len(st.session_state.ranked_candidates) if st.session_state.ranked_candidates else 0)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Job Description", "📄 Upload Resumes", "🏆 Results"])
    
    # Tab 1: Job Description
    with tab1:
        st.header("📝 Job Description")
        st.write("Paste the job description that candidates will be evaluated against:")
        
        job_description = st.text_area(
            "Job Description",
            value=st.session_state.job_description,
            height=300,
            placeholder="""Example:
We are looking for a Senior Python Developer with experience in:
- 5+ years of Python development
- Experience with Django/Flask frameworks  
- Knowledge of databases (PostgreSQL, MongoDB)
- Understanding of REST APIs and microservices
- Experience with cloud platforms (AWS, GCP, Azure)
- Strong problem-solving skills
- Bachelor's degree in Computer Science or related field
"""
        )
        
        if st.button("💾 Save Job Description", type="primary"):
            st.session_state.job_description = job_description
            st.success("✅ Job description saved successfully!")
    
    # Tab 2: Resume Upload
    with tab2:
        st.header("📄 Upload Candidate Resumes")
        
        if not st.session_state.job_description:
            st.warning("⚠️ Please add a job description first in the 'Job Description' tab")
            return
        
        # Rate limiting advice
        st.markdown("""
        <div class="rate-limit-info">
            <strong>💡 Tips for Best Results:</strong><br>
            • Upload 2-5 resumes at a time for optimal processing<br>
            • Larger batches will be automatically split and processed with delays<br>
            • The system includes built-in rate limit protection<br>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT. Recommended: 2-5 files per batch"
        )
        
        if uploaded_files:
            file_count = len(uploaded_files)
            st.write(f"📁 **{file_count} files uploaded**")
            
            if file_count > 5:
                st.info(f"ℹ️ You've uploaded {file_count} files. They will be processed in batches of {BATCH_SIZE} with automatic delays to respect rate limits.")
            
            # Display uploaded files
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size} bytes)")
        
        # Process button
        if st.button("🚀 Process Resumes", type="primary", disabled=not uploaded_files):
            if not groq_api_key:
                st.error("Please provide GROQ API key")
                return
                
            with st.spinner("🔄 Processing resumes with rate limit protection... This may take a few minutes..."):
                try:
                    # Initialize the crew
                    crew = HireWithAICrew(groq_api_key)
                    
                    # Extract text from uploaded files
                    resumes_data = []
                    processor = ResumeProcessor()
                    
                    for uploaded_file in uploaded_files:
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        if file_extension == 'pdf':
                            text = processor.extract_text_from_pdf(uploaded_file)
                        elif file_extension == 'docx':
                            text = processor.extract_text_from_docx(uploaded_file)
                        elif file_extension == 'txt':
                            text = processor.extract_text_from_txt(uploaded_file)
                        else:
                            st.warning(f"Unsupported file format: {uploaded_file.name}")
                            continue
                        
                        if text:
                            resumes_data.append({
                                'filename': uploaded_file.name,
                                'text': text
                            })
                    
                    if not resumes_data:
                        st.error("No valid resumes could be processed")
                        return
                    
                    # Process through AI agents with batching
                    st.info("🤖 Running AI agents with intelligent batching and rate limiting...")
                    results = crew.process_resumes_with_batching(resumes_data, st.session_state.job_description)
                    
                    if results:
                        st.session_state.processed_resumes = results.get('parsed_resumes', [])
                        st.session_state.ranked_candidates = results.get('final_ranking', '')
                        
                        st.success("✅ Resume processing completed successfully!")
                        st.info("📋 Check the 'Results' tab to view the analysis")
                    else:
                        st.error("Failed to process resumes due to rate limits or API issues")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 3: Results  
    with tab3:
        st.header("🏆 Results & Rankings")
        
        if not st.session_state.processed_resumes:
            st.info("📋 No results available. Please process resumes first.")
            return
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Candidate Rankings")
            
            if st.session_state.ranked_candidates:
                st.markdown("### 🥇 Final Rankings")
                st.text_area(
                    "Ranking Results",
                    value=st.session_state.ranked_candidates,
                    height=400
                )
                
                # Download results
                if st.button("💾 Download Results"):
                    results_data = {
                        'job_description': st.session_state.job_description,
                        'processed_resumes': st.session_state.processed_resumes,
                        'final_ranking': st.session_state.ranked_candidates,
                        'timestamp': datetime.now().isoformat(),
                        'model_used': MODEL_DISPLAY_NAME
                    }
                    
                    st.download_button(
                        label="📥 Download Complete Results (JSON)",
                        data=json.dumps(results_data, indent=2),
                        file_name=f"hirewithia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            st.subheader("📈 Summary")
            
            if st.session_state.processed_resumes:
                st.metric("Total Candidates", len(st.session_state.processed_resumes))
                
                summary_data = []
                for resume in st.session_state.processed_resumes:
                    summary_data.append({
                        'Filename': resume['filename'][:20] + "..." if len(resume['filename']) > 20 else resume['filename'],
                        'Status': '✅ Processed'
                    })
                
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
        
        # Individual candidate details
        if st.session_state.processed_resumes:
            st.subheader("📋 Individual Candidate Analysis")
            
            for i, resume in enumerate(st.session_state.processed_resumes):
                with st.expander(f"👤 {resume['filename']}"):
                    st.markdown("**Parsed Data:**")
                    st.text_area(
                        f"Analysis for {resume['filename']}",
                        value=resume['parsed_data'],
                        height=200,
                        key=f"resume_{i}"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🤖 <b>HireWithAI</b> - Powered by CrewAI & GROQ API</p>
        <p><i>Using {MODEL_DISPLAY_NAME} with Rate Limit Protection</i></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
