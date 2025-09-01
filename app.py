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
    from datetime import datetime
    import hashlib
except ImportError as e:
    st.error(f"Missing required dependency: {e}")
    st.stop()

# Configuration
GROQ_MODELS = {
    "llama-3.1-70b-versatile": "Llama 3.1 70B (Recommended)",
    "llama-3.1-8b-instant": "Llama 3.1 8B (Fastest)",
    "mixtral-8x7b-32768": "Mixtral 8x7B"
}

# Initialize session state
if 'processed_resumes' not in st.session_state:
    st.session_state.processed_resumes = []
if 'ranked_candidates' not in st.session_state:
    st.session_state.ranked_candidates = []
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""

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
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_buffer.read())
                tmp_file.flush()
                text = docx2txt.process(tmp_file.name)
                os.unlink(tmp_file.name)  # Clean up temp file
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
    """Main CrewAI multi-agent system for resume screening"""
    
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-70b-versatile"):
        """Initialize the crew with GROQ API"""
        self.llm = LLM(
            model=f"groq/{model}",
            api_key=groq_api_key,
            temperature=0.1
        )
        
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
            goal='Extract structured candidate information from resume text with high accuracy',
            backstory="""You are an expert resume parser with years of experience in 
            extracting structured data from various resume formats. You excel at identifying 
            personal information, work experience, education, skills, and other relevant 
            candidate details from unstructured text.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_skill_matcher_agent(self) -> Agent:
        """Create the Skill Matcher Agent"""
        return Agent(
            role='Skill Matching Expert',
            goal='Match candidate skills with job requirements using advanced NLP techniques',
            backstory="""You are a skilled NLP expert specializing in semantic skill matching. 
            You can identify both explicit and implicit skill matches, understand skill 
            synonyms, and evaluate skill relevance levels. You're excellent at finding 
            transferable skills and assessing skill proficiency levels.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_ranking_agent(self) -> Agent:
        """Create the Ranking Agent"""
        return Agent(
            role='Candidate Ranking Analyst',
            goal='Rank candidates based on job fit, experience, and overall suitability',
            backstory="""You are a senior recruitment analyst with expertise in candidate 
            evaluation and ranking. You excel at weighing multiple factors like skills match, 
            experience level, education relevance, and career progression to create accurate 
            candidate rankings for specific roles.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def create_resume_parsing_task(self, resume_text: str, filename: str) -> Task:
        """Create task for parsing resume"""
        return Task(
            description=f"""
            Parse the following resume text and extract structured information:
            
            Resume Text:
            {resume_text}
            
            Extract the following information in JSON format:
            1. Personal Information (name, email, phone, location)
            2. Professional Summary/Objective
            3. Work Experience (company, position, duration, responsibilities)
            4. Education (degree, institution, graduation year)
            5. Skills (technical and soft skills)
            6. Certifications
            7. Projects (if mentioned)
            8. Total years of experience
            
            Filename: {filename}
            
            Ensure the output is valid JSON format with all relevant fields.
            """,
            expected_output="Structured JSON containing all extracted resume information",
            agent=self.create_resume_parser_agent()
        )
    
    def create_skill_matching_task(self, resume_data: str, job_description: str) -> Task:
        """Create task for matching skills"""
        return Task(
            description=f"""
            Analyze the following candidate data against the job description and provide skill matching analysis:
            
            Candidate Data:
            {resume_data}
            
            Job Description:
            {job_description}
            
            Perform the following analysis:
            1. Identify required skills from job description
            2. Extract candidate skills from resume data
            3. Calculate skill match percentage
            4. Identify missing critical skills
            5. Find transferable/related skills
            6. Assess experience level match
            7. Provide skill gap analysis
            
            Return results in JSON format with match scores and detailed analysis.
            """,
            expected_output="Comprehensive skill matching analysis in JSON format",
            agent=self.create_skill_matcher_agent()
        )
    
    def create_ranking_task(self, candidates_analysis: List[str], job_description: str) -> Task:
        """Create task for ranking candidates"""
        return Task(
            description=f"""
            Rank the following candidates based on their suitability for the job:
            
            Job Description:
            {job_description}
            
            Candidates Analysis:
            {json.dumps(candidates_analysis, indent=2)}
            
            Ranking Criteria:
            1. Skills match percentage (40% weight)
            2. Years of relevant experience (25% weight)
            3. Education relevance (15% weight)
            4. Career progression (10% weight)
            5. Cultural fit indicators (10% weight)
            
            Provide:
            1. Overall ranking with scores
            2. Detailed justification for each candidate
            3. Top 3 recommendations with reasoning
            4. Interview recommendations for top candidates
            
            Return results in JSON format with rankings and explanations.
            """,
            expected_output="Comprehensive candidate ranking with detailed analysis in JSON format",
            agent=self.create_ranking_agent()
        )
    
    def process_resumes(self, resumes_data: List[Dict], job_description: str) -> Dict:
        """Process multiple resumes through the crew"""
        try:
            # Step 1: Parse all resumes
            parsed_resumes = []
            
            for resume_data in resumes_data:
                parsing_task = self.create_resume_parsing_task(
                    resume_data['text'], 
                    resume_data['filename']
                )
                
                parsing_crew = Crew(
                    agents=[self.create_resume_parser_agent()],
                    tasks=[parsing_task],
                    verbose=True
                )
                
                result = parsing_crew.kickoff()
                parsed_resumes.append({
                    'filename': resume_data['filename'],
                    'parsed_data': result.raw,
                    'original_text': resume_data['text']
                })
            
            # Step 2: Match skills for each candidate
            skill_analysis = []
            
            for resume in parsed_resumes:
                skill_task = self.create_skill_matching_task(
                    resume['parsed_data'],
                    job_description
                )
                
                skill_crew = Crew(
                    agents=[self.create_skill_matcher_agent()],
                    tasks=[skill_task],
                    verbose=True
                )
                
                result = skill_crew.kickoff()
                skill_analysis.append({
                    'filename': resume['filename'],
                    'skill_analysis': result.raw,
                    'parsed_data': resume['parsed_data']
                })
            
            # Step 3: Rank all candidates
            ranking_task = self.create_ranking_task(skill_analysis, job_description)
            
            ranking_crew = Crew(
                agents=[self.create_ranking_agent()],
                tasks=[ranking_task],
                verbose=True
            )
            
            ranking_result = ranking_crew.kickoff()
            
            return {
                'parsed_resumes': parsed_resumes,
                'skill_analysis': skill_analysis,
                'final_ranking': ranking_result.raw
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
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
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
        
        # Model selection
        selected_model = st.selectbox(
            "Select GROQ Model",
            options=list(GROQ_MODELS.keys()),
            format_func=lambda x: GROQ_MODELS[x],
            index=0
        )
        
        st.info(f"**Using:** {GROQ_MODELS[selected_model]}")
        
        # Processing statistics
        st.header("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Resumes Processed", len(st.session_state.processed_resumes))
        with col2:
            st.metric("Candidates Ranked", len(st.session_state.ranked_candidates))
    
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
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_files:
            st.write(f"📁 **{len(uploaded_files)} files uploaded**")
            
            # Display uploaded files
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size} bytes)")
        
        # Process button
        if st.button("🚀 Process Resumes", type="primary", disabled=not uploaded_files):
            if not groq_api_key:
                st.error("Please provide GROQ API key")
                return
                
            with st.spinner("🔄 Processing resumes... This may take a few minutes..."):
                try:
                    # Initialize the crew
                    crew = HireWithAICrew(groq_api_key, selected_model)
                    
                    # Extract text from uploaded files
                    resumes_data = []
                    processor = ResumeProcessor()
                    
                    progress_bar = st.progress(0)
                    total_files = len(uploaded_files)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
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
                        
                        progress_bar.progress((i + 1) / total_files)
                    
                    if not resumes_data:
                        st.error("No valid resumes could be processed")
                        return
                    
                    # Process through AI agents
                    st.info("🤖 Running AI agents for resume analysis...")
                    results = crew.process_resumes(resumes_data, st.session_state.job_description)
                    
                    if results:
                        st.session_state.processed_resumes = results.get('parsed_resumes', [])
                        st.session_state.ranked_candidates = results.get('final_ranking', '')
                        
                        st.success("✅ Resume processing completed successfully!")
                        st.info("📋 Check the 'Results' tab to view the analysis")
                    else:
                        st.error("Failed to process resumes")
                
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
                # Display ranking results
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
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        label="📥 Download Complete Results (JSON)",
                        data=json.dumps(results_data, indent=2),
                        file_name=f"hirewithia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            st.subheader("📈 Summary")
            
            # Display summary metrics
            if st.session_state.processed_resumes:
                st.metric("Total Candidates", len(st.session_state.processed_resumes))
                
                # Create summary DataFrame
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
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>🤖 <b>HireWithAI</b> - Powered by CrewAI & GROQ API</p>
        <p><i>Intelligent Multi-Agent Resume Screening System</i></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



