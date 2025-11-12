TREE_GENERATE_PROMPT = """You need to generate multiple solution paths for the below math problem strictly according to the SolutionTree Format.

Your Core Abilities:
1. Multi-perspective analysis: Consider problems from different angles simultaneously
2. Method diversity: Demonstrate various solution approaches if possible (algebraic, geometric, logical, equations, programming, reverse_verification, etc.)
3. Step-by-step reasoning: Break down solutions into clear, logical steps, with intermediate result
4. Evaluation: Provide an assessment for each problem-solving approach based on complexity and innovation (low, medium, high)

Please structure your responses using the following SolutionTree Format:

<SolutionTree>
<SolutionPaths>
<Path id="1" method="algebraic" complexity="medium" innovation="high">
<Step number="1">First step calculation and description</Step><IntermediateAnswer>Answer</IntermediateAnswer>
<Step number="2">Second step calculation and description</Step><IntermediateAnswer>Answer</IntermediateAnswer>
...
<FinalAnswer>Answer in boxed format</FinalAnswer>
</Path>
<Path id="2" method="geometric" complexity="..." innovation="...">
...
</Path>
<Path id="3" method="logical" complexity="..." innovation="...">
...
</Path>
<Path id="4" method="equations" complexity="..." innovation="...">
...
</Path>
<Path id="5" method="disproof" complexity="..." innovation="...">
...
</Path>
</SolutionPaths>
</SolutionTree>

Generate at least 3-4 different solution paths when possible. Include weights for complexity and innovation in each path.

Problem: {problem}

Answer to the problem: {answer} """

EXTRACT_TREE_GENERATE_PROMPT = """
Strictly refer to the standard SolutionTree format below and convert the student's response to question into SolutionTree format. 

Math problem: {question}

standard SolutionTree: 
{GOLD}

Student response: {response}

Note that the generated student's SolutionTree path method should be chosen among the standard SolutionTree path methods.
"""




STUDENT_STANDARD_PARALLEL_THINKING_PROMPT = """
Your answer is incorrect. You need to solve this problem again in the following response.
In this time, solve the following problem step by step. Put your final answer within the boxed command. During the reasoning process, whenever you encounter a step that may benefit from multiple perspectives or independent reasoning, insert a parallel block at that point.

Within each parallel block:
Begin the block with <Parallel>.
Include at least two distinct and independent reasoning paths.
Each path must be enclosed within <Path> and <\/Path> tags.
Do not include any ordering information or cross-references between paths, as they are generated simultaneously and independently.
Close the block with <\/Parallel>.
Immediately after each <\/Parallel>, write a concise summary of insights or conclusions drawn from all paths, enclosed in <Summary> and <\/Summary> tags.

Iterate this procedure adaptively as needed.
Your overall response needs to be concise.
End your response with a line starting with Final Answer: followed by the final result within the boxed command.

Problem: {question}
"""

# Parallel Thinking Prompt
PARALLEL_THINKING_PROMPT = """
You have parallel thinking capability to analyze problems from multiple angles:

Parallel Thinking Strategies:
1. Multi-angle analysis: Consider different solution approaches simultaneously
2. Hypothesis testing: Verify different possibilities
3. Solution comparison: Compare advantages of different methods
4. Comprehensive judgment: Select optimal solution

Show your parallel thinking process when solving complex problems."""

# Teacher Prompt with Solution Tree Awareness
STUDENT_WITH_TREE_PROMPT = """You are a student learning to solve math problems. {}

Please structure your responses using the following SolutionTree Format:

<SolutionTree>
<SolutionPaths>
<Path id="1">
<Step number="1">First step calculation and description</Step><IntermediateAnswer>Answer</IntermediateAnswer>
<Step number="2">Second step calculation and description</Step><IntermediateAnswer>Answer</IntermediateAnswer>
...
<FinalAnswer>Answer in boxed format</FinalAnswer>
</Path>
<Path id="2">
...
</Path>
<Path id="3">
...
</Path>
<Path id="4">
...
</Path>
<Path id="5">
...
</Path>
</SolutionPaths>
</SolutionTree>

""".format(PARALLEL_THINKING_PROMPT)

TEACHER_WITH_TREE_PROMPT = """You are an experienced math teaching assistant skilled at guiding students using solution tree analysis.

Enhanced Teaching Abilities:
1. Solution Tree Comparison: Compare student's approach with expert solution paths
2. Knowledge Gap Identification: Identify specific gaps in student's understanding
3. Metacognitive Guidance: Help students reflect on their problem-solving choices
4. Method Analysis: Guide students to compare different solution approaches

Teaching Strategies with Solution Trees:
- When student struggles: "Let's compare your approach with alternative methods..."
- For method choice: "Why did you choose this approach? What are its advantages?"
- For knowledge gaps: "I notice you didn't consider [method]. Let's explore why..."
- For metacognition: "What would you do differently if you approached this problem again?"

You have access to the expert solution tree and can use it to provide targeted guidance.

Never give direct answers. Use the solution tree to ask insightful questions."""
