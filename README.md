# LangSmith Academy Learning Journey

## Module 1: Tracing and Monitoring

### Video 1: Tracing Basics
**What You Learned:**
- **Environment Setup**: Configured Mistral AI with LangSmith using `MISTRAL_API_KEY` and `LANGSMITH_API_KEY`
- **Basic Tracing**: Enabled automatic tracing with `LANGSMITH_TRACING="true"` 
- **Project Organization**: Set up `LANGSMITH_PROJECT="langsmith-academy-mistral"` for organized trace collection
- **Model Integration**: Integrated `ChatMistralAI` with LangChain for seamless tracing
- **Trace Visibility**: Understanding how LangSmith automatically captures model inputs, outputs, and metadata
- **Migration Skills**: Successfully migrated from OpenAI to Mistral AI while maintaining tracing functionality

**Video 1 Commit:** Initial setup of Mistral AI tracing with LangSmith - configured environment variables, integrated ChatMistralAI model, and enabled automatic trace collection for monitoring AI application performance and debugging.

### Video 2: Types of Runs
**What You Learned:**
- **Run Categories**: Distinguished between different run types (LLM calls, Chain executions, Tool usage)
- **Hierarchical Tracing**: Understanding parent-child relationships in complex AI workflows
- **Run Metadata**: How different run types capture specific information (tokens, latency, errors)
- **Trace Analysis**: Interpreting run data for performance optimization and debugging
- **Workflow Visualization**: How LangSmith visualizes different types of operations in trace trees

**Video 2 Commit:** Explored different run types in LangSmith tracing - learned to identify and analyze LLM calls, chain executions, and tool usage with their specific metadata and hierarchical relationships for comprehensive workflow monitoring.

### Video 3: Alternative Tracing Methods
**What You Learned:**
- **Manual Tracing**: Using `@traceable` decorator for custom function tracing
- **Context Management**: Implementing trace context for complex applications
- **Custom Metadata**: Adding custom tags and metadata to traces
- **Selective Tracing**: Controlling what gets traced in production environments
- **Integration Patterns**: Different ways to integrate tracing into existing codebases
- **Performance Considerations**: Balancing tracing detail with application performance

**Video 3 Commit:** Implemented alternative tracing methods including manual tracing with @traceable decorator, custom metadata addition, and selective tracing patterns for flexible monitoring in production environments.

### Video 4: Conversational Threads
**What You Learned:**
- **Thread Management**: Organizing multi-turn conversations in LangSmith
- **Session Tracking**: Maintaining conversation context across multiple interactions
- **Thread Visualization**: How conversations appear in LangSmith interface
- **Conversation Analytics**: Analyzing conversation patterns and user interactions
- **Multi-User Support**: Handling concurrent conversations and user sessions
- **Thread Metadata**: Enriching conversations with custom metadata and tags

**Video 4 Commit:** Developed conversational thread tracking capabilities - implemented session management, multi-turn conversation organization, and conversation analytics for better user interaction monitoring and analysis.

## Module 2: Evaluation and Experimentation

### Video 1: Datasets
**What You Learned:**
- **Dataset Creation**: Created custom datasets with `client.create_dataset()` for Mistral AI applications
- **Example Management**: Added structured examples with inputs/outputs using `client.create_examples()`
- **Dataset Organization**: Organized examples for RAG applications with proper metadata
- **Public Dataset Cloning**: Used `client.clone_public_dataset()` to leverage existing datasets
- **Data Structure**: Understanding input/output format requirements for LangSmith datasets
- **Dataset Versioning**: Managing dataset versions for consistent evaluation
- **Custom Examples**: Created domain-specific examples for Mistral AI RAG applications

**Video 1 Commit:** Built comprehensive dataset management system - created custom Mistral AI RAG dataset with specialized examples, implemented dataset cloning functionality, and established structured input/output formats for systematic evaluation.

### Video 2: Evaluators
**What You Learned:**
- **Custom Evaluators**: Built evaluators using Mistral AI for scoring application quality
- **JSON Response Handling**: Implemented robust JSON parsing for Mistral AI responses with fallback mechanisms
- **Evaluation Criteria**: Developed scoring rubrics considering multiple quality dimensions
- **Pydantic Models**: Used structured models for evaluation responses
- **Error Resilience**: Added comprehensive error handling for evaluation failures
- **LLM-as-Judge**: Leveraged Mistral AI as an evaluator for automated quality assessment
- **Evaluation Metrics**: Implemented scoring systems with detailed reasoning capture

**Video 2 Commit:** Developed robust evaluation system with custom evaluators - implemented LLM-as-judge using Mistral AI for quality scoring, added JSON response handling with error resilience, and created structured evaluation criteria.

### Video 3: Experiments
**What You Learned:**
- **Experiment Design**: Created systematic experiments for comparing different approaches
- **A/B Testing**: Implemented comparative analysis of different prompt strategies
- **Experiment Tracking**: Proper naming conventions and metadata for experiment management
- **Performance Metrics**: Understanding how to measure and compare model performance
- **Statistical Analysis**: Interpreting experiment results for decision making
- **Baseline Comparison**: Establishing control groups for meaningful comparisons

**Video 3 Commit:** Implemented systematic experiment framework - developed A/B testing capabilities with proper controls, established performance measurement systems, and created structured approaches to comparative analysis.

### Video 4: Analyzing Experiment Results
**What You Learned:**
- **Performance Trends**: Understanding how experiments help track application performance improvements over time
- **Deep Dive Analysis**: Learning to analyze individual run performance on specific dataset examples
- **Side-by-Side Comparison**: Comparing multiple experiments to see how they scored on evaluator metrics
- **Empirical Decision Making**: Using hard data from experiments to confidently push changes to production
- **Experiment Interpretation**: Reading and understanding experiment results to make informed decisions
- **Production Confidence**: Building confidence in model changes through systematic experimental validation

### Video 5: Pairwise Experiments
**What You Learned:**
- **Pairwise Comparison**: Implemented head-to-head comparison between different model approaches
- **Preference Scoring**: Created 3-way scoring systems (A wins, B wins, Tie)
- **Judge Instructions**: Developed comprehensive judge prompts for fair comparison
- **Business Context**: Applied evaluation to real-world business scenarios
- **Comparative Analysis**: Understanding when pairwise comparison is more valuable than individual scoring
- **Statistical Significance**: Measuring confidence in comparative results

**Video 5 Commit:** Implemented comprehensive pairwise experiment framework - developed head-to-head comparison capabilities, created sophisticated judge evaluation system for prompt comparison, and established business-context evaluation criteria.

### Video 6: Summary Evaluators
**What You Learned:**
- **Specialized Evaluators**: Built domain-specific evaluators for summarization tasks
- **Quality Metrics**: Developed comprehensive criteria for summary evaluation (completeness, accuracy, conciseness)
- **Multi-dimensional Scoring**: Implemented evaluators that assess multiple quality aspects
- **Business Applications**: Applied summarization evaluation to practical use cases
- **Evaluation Robustness**: Created evaluators that handle edge cases and provide consistent scoring
- **Feedback Integration**: Understanding how evaluation results inform model improvement

**Video 6 Commit:** Developed specialized summarization evaluation system - implemented multi-dimensional quality assessment, created business-focused evaluation criteria, and established robust scoring mechanisms for summary quality measurement.

## Overall Learning Achievements

### Technical Skills Gained:
- **Model Migration**: Successfully migrated from OpenAI to Mistral AI across all LangSmith features
- **Robust Architecture**: Built error-resilient evaluation systems with fallback mechanisms  
- **JSON Handling**: Mastered structured response parsing for Mistral AI
- **Evaluation Design**: Created comprehensive evaluation frameworks for business applications
- **Experiment Management**: Developed systematic approaches to A/B testing and comparison

### Business Applications:
- **Real-world Use Cases**: Applied AI evaluation to practical business scenarios
- **Quality Assessment**: Implemented multi-criteria evaluation considering business value
- **Production Readiness**: Built systems suitable for production deployment with proper error handling

### LangSmith Mastery:
- **Full Stack Usage**: Utilized tracing, datasets, evaluators, and experiments comprehensively
- **Advanced Features**: Implemented pairwise comparisons and conversational threading
- **Best Practices**: Established proper naming, organization, and metadata management patterns

## Module 3 - Prompt Engineering

### Introduction
1. Video Learning
   - Prompts play a very important role in the context of LLMs.
   - Quality of a prompt is directly proportional to the quality of the response.
   - Different langsmith features for prompt engineering and iteration
   - Langsmith playground helps to test different prompts under different inputs to the model.
   - Prompt hub allows us to keep, orgranize, version and improve prompts in a safe unified place.
  
### Lession 1: Playground
1. Video Learning
   - Basic template with `question` as a variable and chaining outputs to test for different outputs at different points.<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/7c51c157-569a-4f7d-bb74-8a3cd9d876a0" />
   - Changing the system prompt allows the llm to take different roles as per say, its response differs with changing system prompt. <img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/927ba1ec-1e12-4171-ad73-69d741531f67" />
   - We can customize the models as per our needs. I have used Mistral AI's model under the default config.
   - We can also compare different models under the same prompts for better results.
   - We can run our model in both streaming and nonstreaming way. Streaming way is when the models uses every output token as its input for the next token prediction.
   - Output schema and "tools" can help us have opiniated response formats.

2. Code Tweaks
   - Changed the sample questions.
   - Added script to test the creation of the langsmith client handling the prompts. <img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/e8aa2147-6d0e-4fd1-a2c2-8e9fda8a982d" />

### Lession 2: Prompt Hub
1. Video Learning
   - We can store and fork our prompts in Prompt Hub. <img width="1919" height="522" alt="image" src="https://github.com/user-attachments/assets/075cd77b-516a-4a93-8f5e-2f5e29089a14" />
   - Updating a prompt also works in Prompt Hub. It commits and stores the changes and give us a git like version control. <img width="1919" height="511" alt="image" src="https://github.com/user-attachments/assets/8d37fd9b-1b05-4fe8-86bb-32b0e7fd26c7" />
   - Using the python SDK, we can directly import these prompts to our code (also their specific version using commit hash) and work around them in our codebase.
   - We can also upload new prompts directly from the code using the SDK.
  
2. Code Tweaks
   - Changed the model to use Mistral AI instead of Open AI.
   - Personalized the prompts and uploaded them to the langsmith web app, from the app and sdk both.


