"""
Prompt Library for Research Paper Understanding

A modular collection of prompts designed for the PaperCopilot Research Assistant.
Each prompt is optimized for understanding papers from different perspectives:
- Student perspective (intuitive explanations)
- Researcher perspective (deep analysis)
- Reviewer perspective (critical evaluation)
- General question answering with context grounding

All prompts are designed to work with Retrieval-Augmented Generation (RAG),
prioritizing faithfulness to retrieved context over hallucination.

Design Philosophy:
- Modularity: Each prompt is independent and reusable
- Clarity: Prompts explicitly state what to do and what NOT to do
- Context-grounding: All prompts emphasize using only provided context
- Depth over speed: Prompts encourage understanding vs. surface-level summaries
"""

from typing import Optional


def get_student_summary_prompt() -> str:
    """
    Generate a prompt for explaining a research paper to undergraduate students.

    Purpose:
        Translate technical papers into accessible language for students without
        deep expertise in the domain. Focus on intuition and big-picture concepts.

    Output includes:
        - What problem is being solved?
        - Main idea (in simple terms)
        - Method overview (without heavy math)
        - Results (what improved?)
        - Key takeaway (why does this matter?)

    Returns:
        String containing the student-friendly explanation prompt.

    Example:
        >>> prompt = get_student_summary_prompt()
        >>> # Use with LLM: llm.generate(prompt, context=paper_chunks)
    """
    return """You are explaining a research paper to an undergraduate computer science student.

Your goal: Make the paper accessible and engaging, focusing on intuition over rigor.

Based on the provided paper content, create an explanation that includes:

1. **The Problem**
   - What real-world or research challenge does this paper address?
   - Why should anyone care?
   - Use an analogy if helpful.

2. **The Big Idea**
   - What is the main innovation or insight?
   - Explain in 2-3 sentences without heavy math.
   - Focus on the intuition.

3. **How It Works**
   - High-level overview of the method (not detailed algorithms).
   - What are the main steps?
   - Use simple language; avoid jargon.

4. **The Results**
   - What did they achieve?
   - What metrics improved and by how much?
   - How does this compare to previous work?

5. **Why It Matters**
   - What is the broader impact?
   - What can this be used for?
   - What does this enable going forward?

IMPORTANT:
- Use simple analogies and real-world examples.
- Avoid mathematical notation; explain concepts verbally.
- Focus on the "why" not the "how."
- Keep each section 2-3 sentences.
- Be encouraging; show why the paper is cool, not intimidating."""


def get_research_mentor_prompt() -> str:
    """
    Generate a prompt for deep research-level paper analysis.

    Purpose:
        Act as a senior PhD student mentoring a junior researcher. Go beyond
        surface understanding to explore methodology, assumptions, and rigor.

    Output includes:
        - Motivation and positioning
        - Key assumptions and their implications
        - Methodology and technical soundness
        - Strengths of the work
        - Weaknesses and potential issues
        - Likely reviewer criticisms

    Returns:
        String containing the mentor-level analysis prompt.
    """
    return """You are a senior PhD student mentoring a junior researcher who is reading this paper.

Your goal: Develop deep understanding and critical thinking. Push beyond surface-level reading.

Based on the provided paper content, provide a mentor-level analysis with:

1. **Motivation & Positioning**
   - What is the precise research question?
   - How does this fit into the broader field?
   - What gap in existing work does this fill?
   - Is the motivation compelling?

2. **Key Assumptions**
   - What are the underlying assumptions?
   - Which assumptions are stated vs. implicit?
   - What are the implications if these assumptions break?
   - Are assumptions reasonable?

3. **Technical Depth**
   - What is the core technical contribution?
   - Is the methodology sound?
   - Are there any technical flaws or shortcuts?
   - What is novel vs. standard practice?

4. **Strengths of the Work**
   - What is done well?
   - What makes this better than prior work?
   - What insights will stick with you?
   - What did you learn from this paper?

5. **Weaknesses & Concerns**
   - What are the limitations (stated or unstated)?
   - What experiments are missing?
   - What edge cases are not covered?
   - What could break this approach?
   - Is the scope too narrow or too broad?

6. **Anticipated Reviewer Feedback**
   - What would a peer reviewer criticize?
   - Are the claims well-supported?
   - Are there reproducibility concerns?
   - What would strengthen the paper?

7. **Your Research Takeaway**
   - How does this influence your research direction?
   - What techniques could you adapt?
   - What problems does this open up?
   - What would be a natural follow-up?

IMPORTANT:
- Think critically; don't accept claims at face value.
- Ask "why?" and "what if?" repeatedly.
- Focus on building intuition, not just understanding results.
- Cite specific sections when making points."""


def get_reviewer_prompt() -> str:
    """
    Generate a prompt for peer review-level analysis.

    Purpose:
        Analyze the paper as a reviewer for a top-tier venue (NeurIPS, ICLR, ICML).
        Evaluate novelty, experimental rigor, and contribution quality.

    Output includes:
        - Novelty assessment
        - Experimental quality
        - Strength of claims
        - Missing experiments
        - Weaknesses
        - Suggestions for improvement

    Returns:
        String containing the reviewer perspective prompt.
    """
    return """You are a senior reviewer for NeurIPS/ICLR evaluating this paper.

Your goal: Provide constructive but rigorous feedback on novelty, quality, and impact.

Based on the provided paper content, provide a reviewer assessment with:

1. **Novelty & Originality**
   - Is the idea novel or is it incremental?
   - How much is inherited from prior work vs. new contribution?
   - Are there concurrent or very recent related works?
   - Does this advance the field or just optimize within it?

2. **Experimental Design Quality**
   - Are the experiments well-designed?
   - Are comparisons fair and against strong baselines?
   - Are error bars/confidence intervals provided?
   - Are experiments reproducible (code available)?
   - Do datasets represent realistic use cases?

3. **Strength of Claims**
   - Do results support all claims?
   - Are claims overstated?
   - Are statistical tests appropriate?
   - Are limitations acknowledged?
   - Is the scope clearly defined?

4. **Missing Experiments & Analysis**
   - What ablation studies are missing?
   - Should they test on additional datasets?
   - Are there important edge cases?
   - Should they compare against more baselines?
   - Is sensitivity analysis provided?

5. **Technical Soundness**
   - Are there technical errors or concerns?
   - Is the methodology sound?
   - Are hyperparameters justified or tuned fairly?
   - Are assumptions clearly stated?

6. **Weaknesses & Concerns**
   - What are the major weaknesses?
   - What assumptions might not hold in practice?
   - What are the failure modes?
   - How broadly applicable is the approach?
   - Are there ethical considerations?

7. **Concrete Suggestions**
   - What would strengthen this paper?
   - What experiments would be most valuable?
   - What analysis would resolve concerns?
   - What should be added to the main paper vs. appendix?

8. **Overall Assessment**
   - Is this paper ready for publication?
   - What is the likely impact?
   - Is it a clear accept, marginal accept, marginal reject, or reject?
   - If borderline, what would move the needle?

IMPORTANT:
- Be constructive; show where improvements are needed.
- Distinguish between major and minor issues.
- Acknowledge strengths; don't only focus on weaknesses.
- Use specific examples from the paper.
- Consider what reviewers would realistically ask."""


def get_contributions_prompt() -> str:
    """
    Generate a prompt for extracting main contributions.

    Purpose:
        Clearly identify and articulate the paper's main research contributions.
        Force explicit enumeration of novelty claims.

    Returns:
        String containing the contributions extraction prompt.
    """
    return """Extract and clearly articulate the MAIN contributions of this paper.

Based on the provided paper content, identify and list the primary contributions:

Format your response as:

**Contribution 1: [Title]**
[Clear description of what is novel or improved]
[Why is this important?]

**Contribution 2: [Title]**
[Clear description of what is novel or improved]
[Why is this important?]

**Contribution 3: [Title]** (if applicable)
[Clear description of what is novel or improved]
[Why is this important?]

Guidelines:
- Focus on PRIMARY contributions (typically 2-4).
- Avoid listing incremental technical details.
- Each contribution should be distinct and non-overlapping.
- Be specific: instead of "better performance," say "10% improvement on ImageNet."
- Separate methodological contributions from experimental findings.
- If the paper's contributions are unclear, note this explicitly.

IMPORTANT:
- Use only information from the paper.
- Quote or cite section names where contributions are introduced.
- If the paper doesn't clearly state contributions, infer them from results."""


def get_limitations_prompt() -> str:
    """
    Generate a prompt for identifying limitations and weaknesses.

    Purpose:
        Dig into implicit and explicit limitations. Encourage critical thinking
        about when and where the approach fails.

    Returns:
        String containing the limitations identification prompt.
    """
    return """Identify and articulate the LIMITATIONS of this paper.

Based on the provided paper content, thoroughly explore:

1. **Explicit Limitations**
   - What limitations do the authors acknowledge?
   - Where do they say their approach doesn't work?
   - What scope do they explicitly limit?

2. **Hidden Assumptions**
   - What implicit assumptions underlie the work?
   - When would these assumptions break?
   - What real-world constraints are not modeled?

3. **Failure Cases & Edge Cases**
   - When would this method fail?
   - What types of inputs/scenarios are problematic?
   - Are there adversarial cases?
   - How does performance degrade in rare scenarios?

4. **Scalability Issues**
   - How does the method scale with problem size?
   - Are there memory or computational bottlenecks?
   - Would it work for larger/harder variants?
   - What are practical deployment constraints?

5. **Experimental Limitations**
   - Are the datasets representative?
   - Are there evaluation concerns?
   - Are any important metrics missing?
   - Could results be dataset-specific?

6. **Generalization Concerns**
   - Would this work in other domains?
   - Is it specific to the tested setting?
   - What would limit transferability?
   - Are there distribution shift concerns?

7. **Theoretical Gaps**
   - Are theoretical claims justified?
   - Are there missing proofs or explanations?
   - What is assumed vs. proven?

Format:
- Use bullet points for clarity.
- Be specific with evidence from the paper.
- Distinguish between stated and implied limitations.
- Rate severity (critical vs. minor).

IMPORTANT:
- All papers have limitations; be thorough.
- Don't be overly harsh; focus on realistic concerns.
- Use evidence from the paper to support observations."""


def get_future_work_prompt() -> str:
    """
    Generate a prompt for suggesting future research directions.

    Purpose:
        Identify natural extensions, open questions, and new research opportunities
        inspired by this work.

    Returns:
        String containing the future work suggestion prompt.
    """
    return """Suggest FUTURE RESEARCH DIRECTIONS inspired by this paper.

Based on the provided paper content, propose:

1. **Natural Extensions**
   - What is the obvious next step?
   - How could the method be extended?
   - What assumptions could be relaxed?
   - What variants would be interesting?

2. **Open Questions**
   - What does this paper leave unanswered?
   - What interesting questions does it raise?
   - What "why?" questions remain?
   - What connections could be explored?

3. **Improvements & Optimizations**
   - How could the method be improved?
   - Are there better algorithms or architectures?
   - Could computational efficiency be improved?
   - Are there theoretical tightening opportunities?

4. **Broader Applications**
   - What other domains could this apply to?
   - Are there cross-disciplinary connections?
   - What new problems could this solve?
   - What industries could benefit?

5. **Combining with Other Work**
   - How would this combine with [related area]?
   - What would happen if you merged this with [other technique]?
   - Are there synergies with concurrent work?

6. **Addressing Limitations**
   - How could the identified limitations be overcome?
   - What would make this more general?
   - How could failure cases be handled?
   - What would enable scaling?

7. **Fundamental Research Questions**
   - Does this raise deeper questions about [phenomenon]?
   - What principles does this illustrate?
   - What do we learn about the problem structure?

Format:
- Group related ideas.
- Be creative but grounded in the paper's work.
- Suggest specific research questions.
- Rate feasibility and impact.

IMPORTANT:
- Think beyond incremental tweaks.
- Consider both near-term and long-term opportunities.
- Be specific about HOW to pursue each direction."""


def get_dataset_metrics_prompt() -> str:
    """
    Generate a prompt for extracting evaluation setup details.

    Purpose:
        Systematically extract datasets, metrics, baselines, and evaluation
        settings to understand the experimental scope.

    Returns:
        String containing the dataset/metrics extraction prompt.
    """
    return """Extract EVALUATION DETAILS from this paper.

Based on the provided paper content, systematically document:

1. **Datasets Used**
   - Dataset name and source
   - Task (classification, regression, generation, etc.)
   - Dataset size (number of examples, features, classes)
   - Train/val/test split
   - Any preprocessing or modifications

2. **Metrics Evaluated**
   - Metric name and definition
   - Why is this metric appropriate?
   - How is it computed?
   - What do high/low values mean?

3. **Baseline Methods**
   - Baseline name
   - Citation or reference
   - Brief description (1 sentence)
   - Reported performance

4. **Experimental Setup**
   - Hardware used (GPU, CPU, memory)
   - Training procedure (epochs, learning rate, optimizer)
   - Hyperparameters and tuning method
   - Cross-validation or multiple runs?
   - Random seed handling

5. **Statistical Analysis**
   - Error bars or confidence intervals provided?
   - Statistical significance testing?
   - Multiple runs reported?
   - Variability across seeds or splits?

6. **Datasets Not Included**
   - Are there relevant datasets they didn't use?
   - Why were certain datasets excluded?
   - Would results generalize to other domains?

Format as a table:
| Aspect | Details |
|--------|---------|
| Datasets | [list] |
| Metrics | [list] |
| Baselines | [list] |
| Setup | [details] |

IMPORTANT:
- Be comprehensive; capture all evaluation details.
- Use exact names and numbers from the paper.
- Note when information is missing or unclear.
- Identify potential concerns (e.g., unfair comparisons)."""


def get_equation_explanation_prompt() -> str:
    """
    Generate a prompt for explaining mathematical concepts intuitively.

    Purpose:
        Demystify equations by explaining variables, meaning, and intuition
        without heavy mathematical jargon.

    Returns:
        String containing the equation explanation prompt.
    """
    return """Explain the KEY EQUATIONS in this paper in simple, intuitive terms.

For each important equation, provide:

**Equation: [equation reference/number]**

1. **Variables & Symbols**
   - What does each variable represent? (in plain English)
   - What are the dimensions/units?
   - What is the input and output?

2. **English Translation**
   - Say the equation in plain English, as if explaining to a friend.
   - What is it computing or measuring?
   - Why is this quantity important?

3. **Intuition & Meaning**
   - What is the big picture idea?
   - Why does it make sense?
   - Use an analogy if helpful.
   - What would happen if you changed a term?

4. **Importance**
   - Why is this equation central to the paper?
   - What does it enable?
   - How is it used in the method?

5. **Connection to Previous Equations** (if applicable)
   - How does this build on or relate to other equations?
   - What is different from standard/prior formulations?

Guidelines:
- Avoid mathematical jargon; explain like a teacher, not a textbook.
- Use concrete examples if helpful.
- Explain what each term contributes to the overall meaning.
- Focus on KEY equations only (not every formula).
- Make it accessible to someone without advanced math background.

Format example:
---
**Equation: Loss = L(prediction, target)**

Variables: L is loss function, measures prediction error

English: We're computing how wrong our prediction is.

Intuition: Like a grade on a test - higher loss means more wrong.
---

IMPORTANT:
- Prioritize understanding over rigor.
- Use everyday language and examples.
- If an equation is complex, break it into parts.
- Don't assume advanced math knowledge."""


def get_prerequisites_prompt() -> str:
    """
    Generate a prompt for assessing prerequisites and difficulty.

    Purpose:
        Identify the knowledge required to understand the paper and suggest
        a learning progression.

    Returns:
        String containing the prerequisites assessment prompt.
    """
    return """Assess DIFFICULTY & PREREQUISITES for understanding this paper.

Based on the provided paper content, provide:

1. **Difficulty Rating**
   - Rate difficulty 1-10 (1=introductory, 10=extremely specialized)
   - Justify the rating
   - What makes it hard/easy?

2. **Prerequisites (Essential Knowledge)**
   - What must you know BEFORE reading?
   - What concepts/techniques are assumed?
   - What terminology is used without explanation?
   - List in order of importance

3. **Recommended Learning Order**
   - What should you learn first?
   - What builds on what?
   - What can be learned in parallel?
   - Suggest specific resources if applicable

4. **Concepts Introduced**
   - What new techniques/ideas does it introduce?
   - What is novel vs. standard practice?
   - How does this build on prior work?

5. **Background by Field**
   - For someone from ML: what extra background needed?
   - For someone from NLP: what extra background needed?
   - For someone from Systems: what extra background needed?
   - (Adjust fields based on paper)

6. **Estimated Learning Time**
   - How long to understand (for audience X)?
   - Where do people typically get stuck?
   - What sections can be skipped on first read?

7. **Reading Strategy**
   - What is the best order to read sections?
   - Which parts are most important?
   - Which parts can you skim?
   - Where should you slow down?

Format:
- Difficulty: X/10 because [reasons]
- Prerequisites (ordered): 1. [concept], 2. [concept], ...
- Read in this order: [section order with justification]

IMPORTANT:
- Be honest about difficulty; don't minimize or exaggerate.
- Help readers self-assess if they're ready.
- Suggest concrete resources if beneficial."""


def get_roadmap_prompt() -> str:
    """
    Generate a prompt for creating a reading roadmap.

    Purpose:
        Guide readers through the paper with a structured reading plan that
        balances understanding with efficiency.

    Returns:
        String containing the reading roadmap prompt.
    """
    return """Create a READING ROADMAP for this paper.

Based on the provided paper content, structure a guided reading plan:

1. **Difficulty**
   - Overall difficulty 1-10
   - Progression (intro sections hard/easy, does it get harder?)

2. **MUST KNOW (Critical for understanding)**
   - Key concepts that must be understood first
   - Essential background knowledge
   - Core ideas/notation

3. **READ FIRST (Get the gist quickly)**
   - Sections to read for understanding main contribution
   - These give 80% of value in 20% of time
   - Suggested reading order
   - Approximate time: X minutes

4. **READ NEXT (Build deeper understanding)**
   - Sections that provide details and validation
   - These add crucial nuance and evidence
   - Why each section matters
   - Approximate time: X minutes

5. **ADVANCED TOPICS (Deep dives, optional)**
   - Sections for thorough understanding
   - These are important but not essential
   - For different audience interests
   - Approximate time: X minutes

6. **CAN SKIP (Optional or low value)**
   - What can be skipped on first reading?
   - What is redundant?
   - When might you return to these sections?

7. **Key Figures & Tables**
   - Which figures/tables are most important?
   - What story do they tell?
   - Where are they located?

Roadmap structure:

QUICK UNDERSTANDING (15-20 min):
→ Section X: [why and what to focus on]
→ Section Y: [why and what to focus on]
→ Figure Z: [what it shows]

STANDARD READING (45-60 min):
→ [add these sections]
→ [add these sections]

DEEP DIVE (2+ hours):
→ [theory/proofs/detailed methods]

IMPORTANT:
- Tailor recommendations to different audiences if applicable.
- Be specific with section numbers/names.
- Explain WHY each section matters.
- Help readers adjust based on their goals."""


def get_methodology_prompt() -> str:
    """
    Generate a prompt for explaining the technical methodology.

    Purpose:
        Provide a clear, structured explanation of how the method works, from
        training pipeline to architecture details.

    Returns:
        String containing the methodology explanation prompt.
    """
    return """Explain the METHODOLOGY & TECHNICAL APPROACH in detail.

Based on the provided paper content, structure a comprehensive explanation:

1. **High-Level Overview**
   - What is the overall approach in 2-3 sentences?
   - What problem does it solve?
   - How is it different from prior work?

2. **Training Pipeline**
   - What is the input data?
   - What preprocessing/preparation is done?
   - What is the training objective or loss function?
   - What optimization procedure is used?
   - What is the output/result?
   - Describe the flow: input → processing → output

3. **Architecture/Model Design**
   - What is the overall architecture?
   - What are the main components?
   - How do components interact?
   - Draw a conceptual diagram in words

4. **Key Algorithms**
   - What are the core algorithmic innovations?
   - How does it differ from standard approaches?
   - Are there any novel computational tricks?
   - What makes this approach feasible?

5. **Inference/Deployment**
   - How is the trained model used at test time?
   - What is the inference procedure?
   - What are computational requirements?
   - What are practical considerations?

6. **Implementation Details**
   - What are important hyperparameters?
   - How are they set/tuned?
   - What are implementation challenges?
   - Are there engineering tricks?

7. **Comparison to Alternatives**
   - How is this different from approach X?
   - Why choose this over alternatives?
   - What are trade-offs?

8. **Theoretical Justification**
   - Why should this work?
   - What theory supports it?
   - Are there theoretical guarantees?
   - What are assumptions?

Format:
- Use step-by-step breakdown
- Include conceptual diagrams if helpful
- Provide specific numbers/equations where relevant
- Explain what could go wrong

IMPORTANT:
- Be specific; avoid vague descriptions.
- Make it reproducible; include enough detail.
- Explain both WHAT and WHY.
- Connect methodology to experimental results."""


def get_qa_prompt(question: str) -> str:
    """
    Generate a prompt for general question answering with context grounding.

    Purpose:
        Answer arbitrary questions about the paper using only retrieved context.
        Prioritize faithfulness to avoid hallucination.

    Args:
        question: The user's question about the paper.

    Returns:
        String containing the QA prompt with the user's question.

    Example:
        >>> prompt = get_qa_prompt("What datasets were used?")
        >>> # Use with LLM: llm.generate(prompt, context=retrieved_chunks)
    """
    return f"""Answer the following question about the research paper using ONLY the provided context.

QUESTION: {question}

INSTRUCTIONS:

1. **Answer from Context**
   - Use ONLY information from the provided paper context.
   - Do not use external knowledge.
   - Do not guess or infer beyond what is stated.

2. **If Information is Missing**
   - Explicitly state: "This information is not provided in the paper."
   - Do NOT make up an answer.
   - Do NOT hallucinate details.

3. **Citation**
   - Cite section names when possible: "According to the Methods section..."
   - Reference figure/table numbers if relevant
   - Quote sparingly if the question calls for exact wording

4. **Clarity**
   - Be concise and direct
   - Use simple language
   - Structure your answer clearly (use bullets/numbering if helpful)
   - Avoid unnecessary technical jargon

5. **Nuance**
   - If the paper discusses trade-offs or multiple perspectives, mention them
   - If the question is debatable, present both sides from the paper
   - If the answer is uncertain in the paper, convey that uncertainty

ANSWER THE QUESTION:"""


def get_deep_qa_prompt(question: str) -> str:
    """
    Generate a prompt for deep, explanatory question answering.

    Purpose:
        Provide comprehensive, analytical answers that explore "why" and "how"
        questions deeply using paper context.

    Args:
        question: The user's question requiring deep analysis.

    Returns:
        String containing the deep QA prompt with the user's question.

    Example:
        >>> prompt = get_deep_qa_prompt("Why does this method work better than baselines?")
        >>> # Use with LLM: llm.generate(prompt, context=retrieved_chunks)
    """
    return f"""Provide a DEEP, EXPLANATORY answer to the following question about the paper.

QUESTION: {question}

INSTRUCTIONS:

1. **Foundation: What does the paper say?**
   - Start with explicit statements from the paper
   - Quote or cite specific sections
   - Provide factual grounding

2. **Analysis: Why does it work this way?**
   - Explain the reasoning behind the approach
   - Discuss underlying assumptions
   - Connect ideas to fundamentals

3. **Evidence & Examples**
   - Point to experimental results
   - Reference specific datasets/metrics
   - Use concrete examples from the paper

4. **Failure Cases & Limitations**
   - When does this approach NOT work?
   - What assumptions might break?
   - Are there edge cases?
   - What does the paper acknowledge?

5. **Connections to Broader Context**
   - How does this relate to prior work mentioned?
   - What research directions does it enable?
   - What does this reveal about the problem?

6. **Critical Perspective**
   - Are there unstated assumptions?
   - Could there be alternative explanations?
   - What would strengthen the argument?

FORMAT:
- Use clear sections/headers
- Start with the paper's claims, then analyze
- Support all points with evidence from the paper
- Be explicit about what IS and ISN'T in the paper

IMPORTANT:
- Go beyond surface-level; explore the "why"
- Use ONLY information from the provided paper
- Do NOT hallucinate; say "the paper doesn't address this"
- Help the reader develop deeper understanding

ANSWER THE QUESTION:"""


def get_compare_papers_prompt() -> str:
    """
    Generate a prompt for comparing two research papers.

    Purpose:
        Systematically compare multiple papers across dimensions: goals, methods,
        experiments, and suitability for different use cases.

    Returns:
        String containing the paper comparison prompt.

    Note:
        This prompt is designed for use with context from TWO papers.
        Prepend with: "Here are two papers. Compare them on:"
    """
    return """COMPARE THE TWO PROVIDED PAPERS across these dimensions:

1. **Research Goals**
   - What is each paper trying to achieve?
   - How are the goals similar/different?
   - Which has a broader vs. narrower scope?

2. **Technical Approach**
   - What methodology does each use?
   - What are the core technical innovations?
   - Are the approaches related or orthogonal?
   - Which is more complex?

3. **Datasets & Evaluation**
   - What datasets does each use?
   - Which evaluation is more comprehensive?
   - Do they use the same metrics?
   - Are the experimental settings comparable?

4. **Metrics & Results**
   - What performance does each achieve?
   - Can results be directly compared?
   - Which is better on which metrics?
   - Are improvements significant?

5. **Strengths Comparison**
   - What is each paper's strongest point?
   - Where does each innovate?
   - Which approach is more elegant?
   - Which is more practical?

6. **Weaknesses Comparison**
   - What are each paper's limitations?
   - Which faces more fundamental challenges?
   - Which is more limited in scope?
   - Which would be harder to scale?

7. **Novelty & Originality**
   - Which is more novel?
   - Do they build on each other?
   - Is one primarily extending the other?
   - Are they tackling the same problem differently?

8. **Suitable Use Cases**
   - When would you choose Paper A over Paper B?
   - What applications favor Paper A?
   - What applications favor Paper B?
   - Could they be combined?

COMPARISON TABLE:
| Aspect | Paper A | Paper B | Winner |
|--------|---------|---------|--------|
| Novelty | | | |
| Rigor | | | |
| Practical | | | |
| Impact | | | |

9. **Recommendation**
   - If you had to choose one, which would you use and why?
   - What would you do if you had access to both?
   - Which is more likely to influence future work?

IMPORTANT:
- Be balanced; find legitimate strengths/weaknesses in each
- Be specific with numbers and examples
- Use only information from the papers
- Help the reader understand which is better for THEIR goals"""


# ============================================================================
# Utility Functions
# ============================================================================


def get_all_prompts() -> dict:
    """
    Get all available prompts as a dictionary.

    Returns:
        Dictionary mapping prompt names to prompt functions.

    Example:
        >>> prompts = get_all_prompts()
        >>> for name in prompts:
        ...     print(name)
    """
    return {
        "student_summary": get_student_summary_prompt,
        "research_mentor": get_research_mentor_prompt,
        "reviewer": get_reviewer_prompt,
        "contributions": get_contributions_prompt,
        "limitations": get_limitations_prompt,
        "future_work": get_future_work_prompt,
        "dataset_metrics": get_dataset_metrics_prompt,
        "equation_explanation": get_equation_explanation_prompt,
        "prerequisites": get_prerequisites_prompt,
        "roadmap": get_roadmap_prompt,
        "methodology": get_methodology_prompt,
    }


def list_available_prompts() -> list:
    """
    List the names of all available prompts.

    Returns:
        List of prompt names.

    Example:
        >>> names = list_available_prompts()
        >>> print(names)
    """
    return list(get_all_prompts().keys())


def get_prompt_by_name(name: str) -> Optional[str]:
    """
    Retrieve a prompt by name.

    Args:
        name: Name of the prompt (see list_available_prompts()).

    Returns:
        Prompt string, or None if not found.

    Example:
        >>> prompt = get_prompt_by_name("student_summary")
    """
    prompts_map = {
        "student_summary": get_student_summary_prompt,
        "research_mentor": get_research_mentor_prompt,
        "reviewer": get_reviewer_prompt,
        "contributions": get_contributions_prompt,
        "limitations": get_limitations_prompt,
        "future_work": get_future_work_prompt,
        "dataset_metrics": get_dataset_metrics_prompt,
        "equation_explanation": get_equation_explanation_prompt,
        "prerequisites": get_prerequisites_prompt,
        "roadmap": get_roadmap_prompt,
        "methodology": get_methodology_prompt,
    }
    
    if name not in prompts_map:
        return None
    
    return prompts_map[name]()
