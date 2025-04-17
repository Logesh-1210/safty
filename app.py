from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import folium
import os

app = Flask(__name__)
app.secret_key = 'secret123'

# === DB Setup ===
if not os.path.exists('users.db'):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

# === Load Data and Train Models ===
crime_data = pd.read_csv('crime_data.csv')

label_encoders = {}
model_svm = None
model_kmeans = None

def preprocess(df):
    df = df.dropna()
    for col in ['Location', 'Time', 'CrimeType']:
        df[col] = df[col].astype(str).str.lower().str.strip()
        
        if col not in label_encoders:
            le = LabelEncoder()
            df[col] = df[col].fillna('unknown')
            le.fit(list(df[col].unique()) + ['unknown'])
            df[col] = le.transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            # Replace unseen values with 'unknown'
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
            
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            df[col] = le.transform(df[col])    
           # df[col] = label_encoders[col].transform(df[col])
    return df

def train_models():
    global model_svm, model_kmeans
    df = preprocess(crime_data.copy())
    X = df[['Location', 'Time', 'CrimeType']]
    y = df['Severity'] if 'Severity' in df else df.iloc[:, -1]
    model_svm = SVC()
    model_svm.fit(X, y)
    model_kmeans = KMeans(n_clusters=5, n_init=10)
    model_kmeans.fit(df[['Latitude', 'Longitude']])

def predict_crime(location, time, crime_type):
    df = pd.DataFrame([[location, time, crime_type]], columns=['Location', 'Time', 'CrimeType'])
    df = preprocess(df)
    prediction = model_svm.predict(df)[0]
    return f"Predicted Crime Severity: {prediction}"

def generate_heatmap():
    df = crime_data.copy()
    map_ = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=row['CrimeType'],
            fill=True,
            color='red',
            fill_opacity=0.7
        ).add_to(map_)
    return map_._repr_html_()

train_models()

base_css = """
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    margin: 0;
    padding: 0;
    text-align: center;
}
.card, .form-box {
    background: white;
    margin: 5% auto;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    width: 90%;
    max-width: 400px;
}
input {
    width: 100%;
    padding: 12px;
    margin: 10px 0;
    border-radius: 8px;
    border: 1px solid #ccc;
}
input[type="submit"], .button {
    background-color: #007bff;
    color: white;
    font-weight: bold;
    padding: 12px;
    text-decoration: none;
    display: inline-block;
    border-radius: 8px;
    border: none;
    width: 100%;
}
input[type="submit"]:hover, .button:hover {
    background-color: #0056b3;
}
h2 {
    color: #343a40;
}
h3 {
    color: #dc3545;
}
.map-container {
    width: 90%;
    height: 600px;
    margin: 20px auto;
}
</style>
"""

@app.route('/')
def index():
    return render_template_string(f'''
    <html><head><title>Safety Locator</title>{base_css}</head>
    <body>
        <div class="card">
            <h2>🚨 Safety Locator</h2>
            <p>Crime Rate and Hotspot Prediction System for Women</p>
            <a href="{{{{ url_for('login') }}}}" class="button">Login</a>
            <a href="{{{{ url_for('register') }}}}" class="button" style="background-color:#28a745;">Register</a>
        </div>
    </body></html>
    ''')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, p))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except:
            return "Username already exists."
    return render_template_string(f'''
    <html><head><title>Register</title>{base_css}</head>
    <body>
    <div class="form-box">
        <h2>Create Account</h2>
        <form method="post">
            <input name="username" placeholder="Choose a username" required>
            <input name="password" type="password" placeholder="Create password" required>
            <input type="submit" value="Register">
        </form>
        <p><a href="{{{{ url_for('login') }}}}" class="button" style="background-color:#6c757d;">Already have an account?</a></p>
    </div></body></html>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = u
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials."
    return render_template_string(f'''
    <html><head><title>Login</title>{base_css}</head>
    <body>
    <div class="form-box">
        <h2>Login</h2>
        <form method="post">
            <input name="username" placeholder="Username" required>
            <input name="password" type="password" placeholder="Password" required>
            <input type="submit" value="Login">
        </form>
        <p><a href="{{{{ url_for('register') }}}}" class="button" style="background-color:#6c757d;">New user? Register</a></p>
    </div></body></html>
    ''')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    prediction = None
    if request.method == 'POST':
        location = request.form['location']
        time = request.form['time']
        crime_type = request.form['crime_type']
        prediction = predict_crime(location, time, crime_type)

    return render_template_string(f'''
    <html><head><title>Dashboard</title>{base_css}</head>
    <body>
    <div class="form-box">
        <h2>Welcome, {{{{ session['username'] }}}} 👋</h2>
        <form method="post">
            <input name="location" placeholder="Enter Location" required>
            <input name="time" placeholder="Time (e.g. 23:00)" required>
            <input name="crime_type" placeholder="Crime Type" required>
            <input type="submit" value="Predict Crime">
        </form>
        {{% if prediction %}}<h3>{{{{ prediction }}}}</h3>{{% endif %}}
        <a href="{{{{ url_for('heatmap') }}}}" class="button">View Hotspot Map</a>
        <a href="{{{{ url_for('logout') }}}}" class="button" style="background-color:#dc3545;">Logout</a>
    </div></body></html>
    ''', prediction=prediction)

@app.route('/heatmap')
def heatmap():
    if 'username' not in session:
        return redirect(url_for('login'))
    map_html = generate_heatmap()
    return render_template_string(f'''
    <html><head><title>Heatmap</title>{base_css}</head>
    <body>
        <h2>🗺️ Crime Hotspot Heatmap</h2>
        <div class="map-container">{{{{ map_html|safe }}}}</div>
        <a href="{{{{ url_for('dashboard') }}}}" class="button">← Back to Dashboard</a>
    </body></html>
    ''', map_html=map_html)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
