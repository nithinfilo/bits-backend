import os
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal, Summary
import pandas as pd
import tempfile
import traceback
import io
import base64

# Comprehensive matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import *
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.lines as lines
import matplotlib.text as text
import matplotlib.font_manager as font_manager
import matplotlib.backends.backend_agg as backend_agg
import matplotlib.patheffects as patheffects
import matplotlib.collections as collections
import matplotlib.contour as contour
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.legend as legend
import matplotlib.markers as markers
import matplotlib.quiver as quiver
import matplotlib.scale as scale
import matplotlib.spines as spines
import matplotlib.table as table

from openai import OpenAI
import numpy as np
import warnings
from pymongo import MongoClient
from bson import ObjectId
import requests
import threading
import time
import json
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from functools import wraps

# Load environment variables
load_dotenv()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")
warnings.filterwarnings("ignore", category=UserWarning, module="lida.components.summarizer")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
SECRET_KEY = os.getenv("SECRET_KEY")
API_KEY = os.getenv("API_KEY")

# Use /tmp for temporary file storage on Render
TEMP_DIR = '/mnt/temp'

def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        received_key = request.headers.get('Authorization')
        expected_key = f"Bearer {API_KEY}"
        if received_key == expected_key:
            return view_function(*args, **kwargs)
        else:
            return jsonify({"error": "Invalid or missing API key"}), 401
    return decorated_function

if not all([OPENAI_API_KEY, MONGO_URI, SECRET_KEY]):
    raise ValueError("Missing required environment variables")

app.config['SECRET_KEY'] = SECRET_KEY

lida = Manager(text_gen=llm("openai", api_key=OPENAI_API_KEY))
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize MongoDB client after forking (in each request)
def get_db():
    if 'mongo_client' not in g:
        g.mongo_client = MongoClient(MONGO_URI)
    return g.mongo_client.get_database("test")

@app.teardown_appcontext
def close_mongo_connection(exception):
    mongo_client = g.pop('mongo_client', None)
    if mongo_client is not None:
        mongo_client.close()

# Track temp files and their last access time
temp_files = {}

def cleanup_temp_files():
    while True:
        time.sleep(300)  # Check every 5 minutes
        current_time = time.time()
        for file_path, last_accessed in list(temp_files.items()):
            if current_time - last_accessed > 3600:  # 1 hour inactivity
                try:
                    os.unlink(file_path)  # Delete the temp file
                except FileNotFoundError:
                    pass  # File already deleted
                del temp_files[file_path]  # Remove from tracking

# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_temp_files, daemon=True)
cleanup_thread.start()

def get_temp_file_path(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file: HTTP {response.status_code}")
    
    content_type = response.headers.get('content-type', '').lower()
    if 'csv' in content_type:
        suffix = '.csv'
    elif 'json' in content_type:
        suffix = '.json'
    else:
        # Default to CSV if content-type is not explicitly specified
        suffix = '.csv'
    
    os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure the directory exists
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
        temp_files[temp_file_path] = time.time()  # Track the last accessed time
    return temp_file_path

def load_dataset(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            # First, try to read as a regular JSON file
            try:
                df = pd.read_json(file_path)
            except ValueError:
                # If that fails, try to read as JSON Lines
                df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError("Unsupported file format")
        
        app.logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        app.logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_json(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    data = data.replace('NaN', 'null')
    with open(file_path, 'w') as file:
        file.write(data)

def generate_dataset_description(fields):
    fields_info = "\n".join([f"{field['column']}: {field['properties'].get('description', 'No description')}" for field in fields])
    prompt = f"Given a dataset with the following fields:\n{fields_info}\nProvide a brief one-sentence description of what this dataset contains and its potential use."
    
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst providing concise dataset descriptions."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating dataset description: {str(e)}")
        return "Dataset description unavailable."

def deduct_credit(user_id):
    db = get_db()
    result = db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$inc": {"credits": -1}}
    )
    if result.modified_count == 0:
        raise ValueError("User not found or insufficient credits")

@app.route('/api/summarize', methods=['POST'])
@require_api_key
def summarize():
    try:
        data = request.json
        url = data['datasetUrl']
        session_id = data.get('sessionId')
        selected_model = data.get('model', 'gpt-3.5-turbo')
        temperature = data.get('temperature', 0.0)

        file_path = get_temp_file_path(url)
        temp_files[file_path] = time.time()  # Update access time

        try:
            df = load_dataset(file_path)
        except Exception as e:
            app.logger.error(f"Error loading dataset: {str(e)}")
            return jsonify({'error': f"Error loading dataset: {str(e)}"}), 400

        # Log the first few rows of the dataframe for debugging
        app.logger.info(f"First few rows of the loaded dataset:\n{df.head().to_string()}")

        textgen_config = TextGenerationConfig(n=1, temperature=temperature, model=selected_model)
        summary = lida.summarize(file_path, summary_method="llm", textgen_config=textgen_config)

        if not summary.get('dataset_description'):
            summary['dataset_description'] = generate_dataset_description(summary.get('fields', []))

        # Store the file path in the summary
        summary['file_path'] = file_path

        if session_id:
            db = get_db()
            db.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"summary": summary}},
                upsert=True
            )

        return jsonify(summary)
    except Exception as e:
        app.logger.error(f"Error in summarize: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/goals', methods=['POST'])
@require_api_key
def generate_goals():
    try:
        data = request.json
        summary = data['summary']
        custom_goal = data.get('customGoal')
        session_id = data.get('sessionId')

        app.logger.info(f"Generating goals for session: {session_id}")
        app.logger.info(f"Summary: {summary}")
        app.logger.info(f"Custom goal: {custom_goal}")

        db = get_db()
        
        # Fetch existing goals
        session = db.sessions.find_one({"_id": ObjectId(session_id)})
        existing_goals = session.get('goals', []) if session else []

        if custom_goal:
            goal = Goal(question=custom_goal, visualization="", rationale="")
            goals = [goal]
            app.logger.info(f"Generated custom goal: {goal}")
        else:
            textgen_config = TextGenerationConfig(n=4, temperature=0.2, model="gpt-3.5-turbo")
            try:
                goals = lida.goals(summary, n=4, textgen_config=textgen_config)
                app.logger.info(f"Generated {len(goals)} goals using LIDA")
            except Exception as lida_error:
                app.logger.error(f"Error generating goals with LIDA: {str(lida_error)}")
                return jsonify({'error': f"Error generating goals: {str(lida_error)}"}), 500

        goals_list = [
            {
                "question": goal.question,
                "visualization": goal.visualization,
                "rationale": goal.rationale
            }
            for goal in goals
        ]

        # Append new goals to existing ones
        updated_goals = existing_goals + goals_list

        if session_id:
            try:
                # Update the session with the combined goals
                db.sessions.update_one(
                    {"_id": ObjectId(session_id)},
                    {"$set": {"goals": updated_goals}},
                    upsert=True
                )
                
                app.logger.info(f"Updated goals in database for session: {session_id}")
                return jsonify(updated_goals)
            except Exception as db_error:
                app.logger.error(f"Error updating goals in database: {str(db_error)}")
                return jsonify({'error': f"Error updating goals in database: {str(db_error)}"}), 500

        return jsonify(updated_goals)
    except Exception as e:
        app.logger.error(f"Unexpected error in generate_goals: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/visualize', methods=['POST'])
@require_api_key
def generate_visualization():
    try:
        data = request.json
        if 'summary' not in data or 'goal' not in data or 'sessionId' not in data:
            return jsonify({'error': 'Missing required fields: summary, goal, or sessionId'}), 400
        
        summary = data['summary']
        goal = data['goal']
        instruction = data.get('instruction')
        session_id = data['sessionId']
        
        app.logger.info(f"Generating visualization for session: {session_id}")
        app.logger.info(f"Goal: {goal}")
        app.logger.info(f"Instruction: {instruction}")

        db = get_db()
        
        # Fetch the session and extract the user ID
        session = db.sessions.find_one({"_id": ObjectId(session_id)})
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        user_id = session.get('userId')
        if not user_id:
            return jsonify({'error': 'User ID not found in session'}), 400

        # Deduct credit
        try:
            deduct_credit(user_id)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        selected_model = data.get('model', 'gpt-3.5-turbo')
        temperature = data.get('temperature', 0.2)

        textgen_config = TextGenerationConfig(n=1, temperature=temperature, model=selected_model)
        
        plt.figure(figsize=(12, 8))
        
        # Load the dataset
        try:
            file_path = summary.get('file_path')
            if not file_path:
                return jsonify({'error': 'File path is missing in the summary'}), 400
            
            app.logger.info(f"Attempting to load dataset from: {file_path}")
            df = load_dataset(file_path)
            
            app.logger.info(f"Dataset loaded. Shape: {df.shape}")
            app.logger.info(f"Columns: {df.columns.tolist()}")
            app.logger.info(f"First few rows:\n{df.head().to_string()}")
        except Exception as e:
            app.logger.error(f"Error loading dataset: {str(e)}")
            return jsonify({'error': f'Error loading dataset: {str(e)}'}), 400

        # Update summary with actual column names
        summary['field_names'] = df.columns.tolist()

        # Create Summary object for LIDA
        summary_obj = Summary(
            name=os.path.basename(file_path),
            file_name=file_path,
            fields=summary.get('fields', []),
            dataset_description=summary.get('dataset_description', ''),
            field_names=summary['field_names']
        )

        goal_obj = Goal(**goal) if isinstance(goal, dict) else goal

        # Fetch existing visualization if available
        existing_visualization = None
        existing_visualizations = session.get('visualizations', [])
        existing_visualization = next((v for v in existing_visualizations if v['goal']['question'] == goal['question']), None)

        if instruction and existing_visualization:
            # Modify existing visualization
            modification_history = existing_visualization.get('modification_history', [])
            
            charts = lida.edit(
                code=existing_visualization['code'],
                summary=summary_obj,
                instructions=[m['instruction'] for m in modification_history] + [instruction],
                library="matplotlib",
                textgen_config=textgen_config
            )
        else:
            # Generate new visualization
            charts = lida.visualize(summary=summary_obj, goal=goal_obj, library="matplotlib", textgen_config=textgen_config)
            modification_history = []

        if not charts:
            return jsonify({'error': 'No visualizations were generated'}), 500
        
        chart = charts[0]
        
        # Prepare local variables for exec()
        local_vars = {
            'data': df,
            'plt': plt,
            'pd': pd,
            'np': np,
            'mpatches': matplotlib.patches,
            'matplotlib': matplotlib,
            'cm': matplotlib.cm,
            'colors': matplotlib.colors,
            'patches': matplotlib.patches,
            'transforms': matplotlib.transforms,
            'ticker': matplotlib.ticker,
            'animation': matplotlib.animation,
            'mpimg': matplotlib.image,
            'lines': matplotlib.lines,
            'text': matplotlib.text,
            'font_manager': matplotlib.font_manager,
            'backend_agg': matplotlib.backends.backend_agg,
            'patheffects': matplotlib.patheffects,
            'collections': matplotlib.collections,
            'contour': matplotlib.contour,
            'mdates': matplotlib.dates,
            'gridspec': matplotlib.gridspec,
            'legend': matplotlib.legend,
            'markers': matplotlib.markers,
            'quiver': matplotlib.quiver,
            'scale': matplotlib.scale,
            'spines': matplotlib.spines,
            'table': matplotlib.table
        }
        
        try:
            exec(chart.code, globals(), local_vars)
        except Exception as e:
            app.logger.error(f"Error executing visualization code: {str(e)}")
            app.logger.error(f"Visualization code:\n{chart.code}")
            return jsonify({'error': f"Error generating visualization: {str(e)}"}), 400
        
        # Convert the chart to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Add the new modification to the history
        if instruction:
            modification_history.append({
                'instruction': instruction,
                'raster': img_base64
            })
        
        response = {
            'code': chart.code,
            'raster': img_base64,
            'goal': goal,
            'modification_history': modification_history
        }

        # Update the session with the new or modified visualization
        if existing_visualization:
            db.sessions.update_one(
                {"_id": ObjectId(session_id), "visualizations.goal.question": goal['question']},
                {"$set": {"visualizations.$": response}},
            )
        else:
            db.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$push": {"visualizations": response}},
                upsert=True
            )

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error in generate_visualization: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# Update the deduct_credit function to handle ObjectId
def deduct_credit(user_id):
    db = get_db()
    result = db.users.update_one(
        {"_id": user_id},  # user_id is already an ObjectId
        {"$inc": {"credits": -1}}
    )
    if result.modified_count == 0:
        raise ValueError("User not found or insufficient credits")

@app.route('/api/sessions/<session_id>/visualizations', methods=['GET'])
@require_api_key
def get_session_visualizations(session_id):
    try:
        db = get_db()
        session = db.sessions.find_one({"_id": ObjectId(session_id)})
        if session and 'visualizations' in session:
            return jsonify(session['visualizations'])
        return jsonify([])
    except Exception as e:
        app.logger.error(f"Error fetching visualizations for session {session_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>/goals', methods=['GET'])
@require_api_key
def get_session_goals(session_id):
    try:
        app.logger.info(f"Fetching goals for session: {session_id}")
        db = get_db()
        session = db.sessions.find_one({"_id": ObjectId(session_id)})
        if session and 'goals' in session:
            app.logger.info(f"Found {len(session['goals'])} goals for session {session_id}")
            return jsonify(session['goals'])
        app.logger.info(f"No goals found for session {session_id}")
        return jsonify([])
    except Exception as e:
        app.logger.error(f"Error fetching goals for session {session_id}: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make the app accessible externally
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)