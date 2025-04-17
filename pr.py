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
@keyframes fadeInUp {
  0% { opacity: 0; transform: translateY(20px); }
  100% { opacity: 1; transform: translateY(0); }
}
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(145deg, #0f2027, #203a43, #2c5364);
    color: #f8f9fa;
    margin: 0;
    padding: 0;
    animation: fadeInUp 1s ease-in-out;
}
.card, .form-box {
    background: rgba(0,0,0,0.8);
    margin: 5% auto;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0 8px 30px rgba(255,0,0,0.3);
    width: 90%;
    max-width: 600px;
    animation: fadeInUp 1.2s ease-in-out;
}
input, select {
    width: 100%;
    padding: 12px;
    margin: 10px 0;
    border-radius: 8px;
    border: 1px solid #444;
    background-color: #222;
    color: #fff;
}
input[type="submit"], .button {
    background-color: #dc3545;
    color: white;
    font-weight: bold;
    padding: 12px;
    text-decoration: none;
    display: inline-block;
    border-radius: 8px;
    border: none;
    width: 100%;
    transition: background 0.3s;
}
input[type="submit"]:hover, .button:hover {
    background-color: #c82333;
}
h2 {
    color: #ffc107;
}
h3 {
    color: #17a2b8;
}
.map-container {
    width: 90%;
    height: 600px;
    margin: 20px auto;
}
table {
    width: 100%;
    margin-top: 20px;
    border-collapse: collapse;
    color: #f8f9fa;
}
th, td {
    padding: 10px;
    border: 1px solid #444;
}
th {
    background-color: #343a40;
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
            <p style="margin-top:20px;">Welcome to the Safety Locator platform. Our system uses advanced machine learning models to predict crime severity and detect high-risk zones for women across various cities.</p>
            
            <h3 style="margin-top:30px;">🔥 What are Crime Hotspots?</h3>
            <p>Crime hotspots are geographical areas with a high concentration of criminal activity. By identifying these zones, we can help citizens stay informed and authorities act faster.</p>
            
            <h3 style="margin-top:30px;">🧩 Common Crime Types in the System</h3>
            <ul>
                <li><b>Harassment</b> – Unwanted behavior, verbal or physical, causing distress or fear.</li>
                <li><b>Assault</b> – Physical attacks or threats of violence.</li>
                <li><b>Molestation</b> – Inappropriate or unwanted physical contact.</li>
                <li><b>Stalking</b> – Repeated, unwanted following or monitoring of someone.</li>
                <li><b>Verbal Abuse</b> – Using harsh or demeaning language to threaten or harm.</li>
            </ul>

            <p style="margin-top:20px;">Together, we can create a safer environment by using technology to stay aware and alert. Let’s begin.</p>

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
sample_data = [
    ("Mumbai", "22:00", "Harassment", 3, 19.0760, 72.8777),
    ("Delhi", "21:30", "Assault", 4, 28.7041, 77.1025),
    ("Chennai", "23:45", "Molestation", 3, 13.0827, 80.2707),
    ("Kolkata", "20:15", "Stalking", 2, 22.5726, 88.3639),
    ("Bangalore", "19:00", "Verbal Abuse", 2, 12.9716, 77.5946),
    ("Hyderabad", "18:30", "Assault", 4, 17.3850, 78.4867),
    ("Ahmedabad", "22:45", "Harassment", 3, 23.0225, 72.5714),
    ("Pune", "21:00", "Molestation", 3, 18.5204, 73.8567),
    ("Jaipur", "20:30", "Stalking", 2, 26.9124, 75.7873),
    ("Lucknow", "23:00", "Verbal Abuse", 2, 26.8467, 80.9462)
]
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
        <h2>Welcome, {{{{ session['username'] }}}} 👮</h2>
        <form method="post">
            <input name="location" placeholder="Enter Location" required>
            <input name="time" placeholder="Time (e.g. 23:00)" required>
            <input name="crime_type" placeholder="Crime Type" required>
            <input type="submit" value="Predict Crime Severity">
        </form>
        {{% if prediction %}}<h3>{{{{ prediction }}}}</h3>{{% endif %}}
        <a href="{{{{ url_for('heatmap') }}}}" class="button">🔍 View Crime Hotspots</a>
        <a href="{{{{ url_for('logout') }}}}" class="button" style="background-color:#6c757d;">Logout</a>
        
        <h2 style="margin-top:40px;">📋 Sample Crime Data</h2>
        <table>
            <tr>
                <th>Location</th><th>Time</th><th>CrimeType</th><th>Severity</th><th>Latitude</th><th>Longitude</th>
            </tr>
            {''.join(f"<tr><td>{l}</td><td>{t}</td><td>{c}</td><td>{s}</td><td>{lat}</td><td>{lon}</td></tr>" for l,t,c,s,lat,lon in sample_data)}
        </table>
    </div>
    </body></html>
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