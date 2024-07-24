import os
import json
from tqdm import tqdm
import argparse
import multiprocessing
from functools import partial

from openagi.agent import Admin
from openagi.llms.azure import AzureChatOpenAIModel
from openagi.memory import Memory
from openagi.planner.task_decomposer import TaskPlanner
from openagi.worker import Worker
from openagi.actions.tools.ddg_search import DuckDuckGoSearch
# from openagi.actions.tools.web_context import WebBaseContextTool

from dotenv import load_dotenv
load_dotenv()

prompt_library = {
    "MCQ": "In this problem, only one option will be correct. Give a detailed solution and end the solution with the final answer.",
    "MCQ(multiple)": "In this problem, multiple options can be correct. Give a detailed solution and end the solution with the final answer.", 
    "Integer": "In this problem, the final answer will be a non-negative integer. Give a detailed solution and end the solution with the final answer.",
    "Numeric": "In this problem, the final will be a numeric value. Give the numerical answer correct up to the 2nd decimal digit. Give a detailed solution and end the solution with the final answer.",
}

def get_response(question, config, mode, response_file, lock):
    llm = AzureChatOpenAIModel(config=config)
    
    researcher = Worker(
        role="Research Assistant",
        instructions="""
        You are a research assistant for solving JEE (Joint Entrance Examination) problems:
        - Search for relevant formulas, concepts, and solved examples
        - Collect information on problem-solving techniques specific to the question type
        - Look for any additional context that might help in solving the problem
        Compile findings to assist the problem solver.
        """,
        actions=[DuckDuckGoSearch],
    )
    
    problem_solver = Worker(
        role="Problem Solver",
        instructions="""
        You are an expert at solving JEE problems:
        - Use the information provided by the research assistant
        - Apply relevant formulas and concepts to solve the problem
        - Show your work step-by-step
        - Clearly state the final answer in the format specified by the question type
        Provide a detailed solution that a student could learn from.
        """,
        actions=[],
    )
    
    solution_reviewer = Worker(
        role="Solution Reviewer",
        instructions="""
        You are a meticulous reviewer of JEE problem solutions:
        - Check the problem solver's work for accuracy
        - Verify that all steps are logical and clearly explained
        - Ensure the final answer matches the question requirements
        - If you find any errors or areas for improvement, provide feedback
        Confirm the final answer or suggest corrections if needed.
        """,
        actions=[DuckDuckGoSearch],
    )

    admin = Admin(
        planner=TaskPlanner(human_intervene=False),
        memory=Memory(),
        llm=llm,
    )
    admin.assign_workers([researcher, problem_solver, solution_reviewer])

    response_dict = question.copy()
    prefix_prompt = prompt_library[question['type']]
    suffix_prompt = ""

    if mode in ['CoT', 'CoT+SC', 'CoT+Exam']:
        suffix_prompt = "Let's think step by step.\n"

    ques = question["question"]
    stripped_ques = ques.replace("\n\n", "\n").strip()
    prompt = prefix_prompt + "\n\n" + "Problem: " + stripped_ques + "\nSolution: " + suffix_prompt

    response_dict["prompt"] = prompt
    print(f'Question: {question["description"]}, Index: {question["index"]}, Mode: {mode}, query begins')

    try:
        response = admin.run(
            query=prompt,
            description=f"""
            Coordinate the team to solve this {question['type']} problem from the JEE:
            1. Have the Research Assistant gather relevant information
            2. Let the Problem Solver use this information to solve the problem step-by-step
            3. Ask the Solution Reviewer to check the work and confirm the final answer
            Ensure the final solution is correct, clear, and educational.
            """,
        )

        response_dict[f"OpenAGI_{mode}_response"] = {'choices': [{'text': response}]}
        
        # Extract the final answer
        extracted_answer = extract_answer(response, question['type'])
        response_dict['extract'] = extracted_answer
        
    except Exception as e:
        print(f"Failure for question {question['index']}!", e)
        response_dict[f"OpenAGI_{mode}_response"] = {'choices': [{'text': 'Error occurred'}]}
        response_dict['extract'] = 'None'

    lock.acquire()
    write_in_file(response_file, response_dict, question, mode)
    lock.release()

    return response_dict

def extract_answer(response, question_type):
    if question_type in ['MCQ', 'MCQ(multiple)']:
        import re
        match = re.search(r'(?:^|\n)(?:Answer:|Final Answer:)?\s*([A-D]+)\s*$', response, re.MULTILINE)
        if match:
            return match.group(1)
    elif question_type in ['Integer', 'Numeric']:
        import re
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]
    return 'None'

def write_in_file(response_file, response_dict, question, mode):
    if os.path.exists(response_file):
        with open(response_file, 'r') as infile:
            responses = json.load(infile)
    else:
        responses = []

    found = False
    for i, old_resp in enumerate(responses):
        if old_resp['description'] == question['description'] and old_resp['index'] == question['index']:
            responses[i] = response_dict  # Replace the entire entry
            found = True
            break

    if not found:
        responses.append(response_dict)
        
    json.dump(sorted(responses, key=lambda elem: (elem['description'], elem['index'])), open(response_file, 'w'), indent=4)
    print(f"####UPDATED {response_file}, Current size: {len(responses)}####")

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--data', default='data/dataset.json')
    args.add_argument('--mode', default='normal')
    args.add_argument('--num_procs', default=1, type=int)
    args.add_argument('--max_questions', default=-1, type=int)
    args = args.parse_args()

    config = AzureChatOpenAIModel.load_from_env_config()
    
    out_file_dir = f'responses/OpenAGI_{args.mode}_responses'
    out_file = os.path.join(out_file_dir, 'responses.json')
    questions = json.load(open(args.data))

    if args.max_questions == -1:
        args.max_questions = len(questions)

    rem_ques = []
    
    if os.path.exists(out_file):
        with open(out_file, 'r') as infile:
            existing_responses = json.load(infile)
        existing_indices = set(resp['index'] for resp in existing_responses)
        
        for question in questions[:args.max_questions]:
            if question['index'] not in existing_indices:
                rem_ques.append(question)
    else:
        os.makedirs(out_file_dir, exist_ok=True)
        rem_ques = questions[:args.max_questions]
    
    print(f"There are {len(rem_ques)} problems remaining")
    
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pool = multiprocessing.Pool(args.num_procs)
    f = partial(get_response, config=config, mode=args.mode, response_file=out_file, lock=lock)
    
    results = list(tqdm(pool.imap(f, rem_ques), total=len(rem_ques)))

    all_responses = existing_responses if os.path.exists(out_file) else []
    for result in results:
        if result:
            all_responses.append(result)
    
    all_responses.sort(key=lambda elem: (elem['description'], elem['index']))
    json.dump(all_responses, open(out_file, 'w'), indent=4)
    print(f"####FINAL UPDATE {out_file}, Total questions: {len(all_responses)}####")

if __name__ == '__main__':
    main()