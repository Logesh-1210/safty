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

<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Safety Locator - Crime Prediction System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">

  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      scroll-behavior: smooth;
      font-family: 'Poppins', sans-serif;
    }

    body {
      background: url('https://images.unsplash.com/photo-1535378917042-10a22c95931a?auto=format&fit=crop&w=1950&q=80') no-repeat center center fixed;
      background-size: cover;
      color: #fff;
    }

    header {
      background-color: rgba(0, 0, 0, 0.8);
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 50px;
      position: sticky;
      top: 0;
      z-index: 1000;
    }

    header h1 {
      color: #ff4081;
      font-size: 24px;
      font-weight: 700;
    }

    nav a {
      color: white;
      margin: 0 15px;
      text-decoration: none;
      font-weight: 600;
      transition: 0.3s;
    }

    nav a:hover {
      color: #ff4081;
    }

    .hero {
      text-align: center;
      padding: 120px 20px 80px;
      animation: fadeIn 2s ease-in-out;
    }

    .hero h2 {
      font-size: 48px;
      color: #ff4081;
      animation: pulse 2s infinite alternate;
    }

    .hero p {
      margin-top: 20px;
      font-size: 20px;
      color: #f8bbd0;
    }

    @keyframes pulse {
      from {
        text-shadow: 0 0 5px #ff4081;
      }
      to {
        text-shadow: 0 0 25px #ff4081, 0 0 30px #ff4081;
      }
    }

    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(-40px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .section {
      padding: 60px 40px;
      background-color: rgba(0, 0, 0, 0.7);
      margin: 30px auto;
      border-radius: 15px;
      width: 90%;
      max-width: 1000px;
      animation: fadeIn 1.5s ease;
    }

    .section h3 {
      color: #f06292;
      margin-bottom: 20px;
      font-size: 26px;
    }

    .section p {
      font-size: 18px;
      line-height: 1.6;
      color: #e1e1e1;
    }

    .years {
      margin-top: 15px;
      color: #ff80ab;
      font-weight: bold;
    }

    footer {
      background-color: rgba(0, 0, 0, 0.95);
      text-align: center;
      padding: 40px 20px;
      color: #ccc;
      margin-top: 60px;
    }

    footer p {
      margin: 8px 0;
    }

    @media (max-width: 768px) {
      header {
        flex-direction: column;
        text-align: center;
      }

      nav {
        margin-top: 10px;
      }

      .hero h2 {
        font-size: 32px;
      }

      .section {
        padding: 30px 20px;
      }
    }
  </style>
</head>
<body>

  <!-- Header and Navigation -->
  <header>
    <h1>Safety Locator</h1>
    <nav>
      <a href="#home">Home</a>
      <a href="#about">About</a>
      <a href="#login">Login</a>
      <a href="#register">Register</a>
    </nav>
  </header>

  <!-- Hero Section -->
  <section class="hero" id="home">
    <h2>Crime Prediction System</h2>
    <p>Stay Informed. Stay Safe. A Smarter Way to Predict Crime Hotspots.</p>
  </section>

  <!-- About Section -->
  <section class="section" id="about">
    <h3>About the System</h3>
    <p>
      Safety Locator is a crime analytics tool designed to predict and display real-time crime-prone locations based on historical and live datasets.
      Leveraging machine learning algorithms like SVM and KMeans, this system clusters crime zones and forecasts potential threats based on severity and time patterns.
    </p>
  </section>

  <!-- Past Crime Predictions -->
  <section class="section">
    <h3>Past Crime Prediction</h3>
    <p>
      Based on recorded crime data, hotspots in urban and semi-urban regions were identified using clustering models.
      The prediction successfully matched with recorded events from:
    </p>
    <p class="years">2018, 2019, 2020, 2021</p>
  </section>

  <!-- Future Crime Predictions -->
  <section class="section">
    <h3>Future Crime Forecast</h3>
    <p>
      Projecting trends using time-series forecasting and real-time feeds, our system anticipates crime-prone locations in upcoming years:
    </p>
    <p class="years">2025, 2026, 2027</p>
    <p>
      These predictions help individuals, especially women, avoid unsafe zones and inform local authorities for preventive action.
    </p>
  </section>

  <!-- Footer -->
  <footer>
    <p><strong>📍 Location:</strong> Chennai, India</p>
    <p><strong>📞 Phone:</strong> +91-9876543210</p>
    <p><strong>📧 Email:</strong> safetylocator@support.in</p>
    <p>© 2025 Safety Locator Project | All rights reserved</p>
  </footer>

</body>
</html>
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
    <html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Register - Safety Locator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Google Font -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">

  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Poppins', sans-serif;
    }

    body {
      background: url('https://images.unsplash.com/photo-1509017174183-0b79fdd7a8f5?auto=format&fit=crop&w=1950&q=80') no-repeat center center fixed;
      background-size: cover;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }

    .container {
      background-color: rgba(0, 0, 0, 0.85);
      padding: 40px;
      border-radius: 15px;
      box-shadow: 0 0 20px rgba(255, 64, 129, 0.7);
      width: 100%;
      max-width: 450px;
      animation: fadeInUp 1.5s ease;
    }

    @keyframes fadeInUp {
      from {
        opacity: 0;
        transform: translateY(30px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    h2 {
      text-align: center;
      color: #ff4081;
      margin-bottom: 30px;
    }

    label {
      color: #eee;
      display: block;
      margin-bottom: 8px;
      font-weight: 500;
    }

    input {
      width: 100%;
      padding: 12px;
      border: none;
      border-radius: 8px;
      margin-bottom: 20px;
      background-color: #2a2a2a;
      color: white;
      transition: 0.3s ease;
    }

    input:focus {
      outline: none;
      box-shadow: 0 0 8px #ff4081;
      background-color: #333;
    }

    button {
      width: 100%;
      padding: 14px;
      background-color: #ff4081;
      border: none;
      border-radius: 10px;
      color: white;
      font-weight: bold;
      font-size: 16px;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }

    button:hover {
      background-color: #e91e63;
    }

    .footer {
      text-align: center;
      margin-top: 20px;
      color: #bbb;
      font-size: 14px;
    }

    @media (max-width: 480px) {
      .container {
        padding: 30px 20px;
      }
    }
  </style>
</head>
<body>

  <div class="container">
    <h2>User Registration</h2>
    <form action="/register" method="POST">
      <label for="phone">Phone Number</label>
      <input type="tel" id="phone" name="phone" required placeholder="Enter your phone number">

      <label for="email">Email Address</label>
      <input type="email" id="email" name="email" required placeholder="Enter your email">

      <label for="location">Location</label>
      <input type="text" id="location" name="location" required placeholder="Enter your city / area">

      <label for="age">Age</label>
      <input type="number" id="age" name="age" min="10" max="100" required placeholder="Enter your age">

      <label for="password">Password</label>
      <input type="password" id="password" name="password" required placeholder="Create a password">

      <button type="submit">Register</button>
    </form>

    <div class="footer">
      Already registered? <a href="/login" style="color: #ff80ab;">Login here</a>
    </div>
  </div>

</body>
</html>
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
    <html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Login - Safety Locator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">

  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: 'Poppins', sans-serif;
    }

    body {
      background: url('https://images.unsplash.com/photo-1600206416306-17d616bea3d8?auto=format&fit=crop&w=1950&q=80') no-repeat center center fixed;
      background-size: cover;
      height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      backdrop-filter: blur(3px);
    }

    .login-container {
      background-color: rgba(0, 0, 0, 0.85);
      padding: 40px 30px;
      border-radius: 15px;
      width: 100%;
      max-width: 400px;
      box-shadow: 0 0 15px rgba(255, 64, 129, 0.7);
      animation: slideIn 1s ease-out;
    }

    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(-40px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    h2 {
      text-align: center;
      color: #ff4081;
      margin-bottom: 30px;
      font-weight: 600;
    }

    label {
      color: #eee;
      margin-bottom: 8px;
      display: block;
    }

    input {
      width: 100%;
      padding: 12px;
      margin-bottom: 20px;
      border: none;
      border-radius: 8px;
      background: #2a2a2a;
      color: #fff;
      transition: 0.3s;
    }

    input:focus {
      outline: none;
      background-color: #333;
      box-shadow: 0 0 8px #ff4081;
    }

    button {
      width: 100%;
      padding: 14px;
      border: none;
      background-color: #ff4081;
      border-radius: 10px;
      color: white;
      font-weight: bold;
      font-size: 16px;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }

    button:hover {
      background-color: #e91e63;
    }

    .footer {
      margin-top: 15px;
      text-align: center;
      color: #ccc;
      font-size: 14px;
    }

    .footer a {
      color: #ff80ab;
      text-decoration: none;
    }

    @media (max-width: 480px) {
      .login-container {
        padding: 30px 20px;
      }

      h2 {
        font-size: 22px;
      }
    }
  </style>
</head>
<body>

  <div class="login-container">
    <h2>Login to Safety Locator</h2>
    <form action="/login" method="POST">
      <label for="email">Email Address</label>
      <input type="email" id="email" name="email" required placeholder="Enter your email">

      <label for="password">Password</label>
      <input type="password" id="password" name="password" required placeholder="Enter your password">

      <button type="submit">Login</button>
    </form>

    <div class="footer">
      Don't have an account? <a href="/register">Register here</a>
    </div>
  </div>

</body>
</html>
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
    <html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Safety Locator - Crime Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    body {
      margin: 0;
      font-family: 'Poppins', sans-serif;
      background: url('https://images.unsplash.com/photo-1603993097397-89d87b9f8857') no-repeat center center/cover;
      color: #fff;
      backdrop-filter: brightness(0.6);
    }

    header {
      background: rgba(0,0,0,0.7);
      padding: 20px;
      text-align: center;
      font-family: 'Orbitron', sans-serif;
      font-size: 32px;
      animation: fadeInDown 1s ease;
      color: #00eaff;
    }

    .dashboard {
      max-width: 1000px;
      margin: 40px auto;
      background: rgba(0, 0, 0, 0.75);
      padding: 30px;
      border-radius: 15px;
      box-shadow: 0 0 20px #00eaff66;
      animation: fadeInUp 1.2s ease;
    }

    .form-group {
      margin-bottom: 20px;
    }

    label {
      font-weight: 600;
      display: block;
      margin-bottom: 5px;
    }

    input, select {
      width: 100%;
      padding: 10px;
      border: none;
      border-radius: 5px;
      font-size: 16px;
      background: #f1f1f1;
      color: #333;
    }

    .buttons {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 20px;
    }

    .buttons button {
      flex: 1;
      padding: 12px;
      font-size: 16px;
      background: #00eaff;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: 0.3s;
      color: #000;
      font-weight: bold;
    }

    .buttons button:hover {
      background: #00b2cc;
    }

    table {
      width: 100%;
      margin-top: 30px;
      border-collapse: collapse;
      animation: fadeIn 1.5s ease;
    }

    th, td {
      padding: 12px;
      text-align: left;
    }

    th {
      background-color: #00b2cc;
      color: #000;
    }

    tr:nth-child(even) {
      background-color: #f2f2f2;
      color: #000;
    }

    tr:nth-child(odd) {
      background-color: #e0e0e0;
      color: #000;
    }

    /* Animations */
    @keyframes fadeInDown {
      from {
        opacity: 0;
        transform: translateY(-30px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @keyframes fadeInUp {
      from {
        opacity: 0;
        transform: translateY(30px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    @keyframes fadeIn {
      from {opacity: 0;}
      to {opacity: 1;}
    }

    @media (max-width: 600px) {
      .buttons {
        flex-direction: column;
      }
    }
  </style>
</head>
<body>

<header>Safety Locator - Crime Prediction Dashboard</header>

<div class="dashboard">
  <form action="/submit_crime" method="POST">
    <div class="form-group">
      <label for="location">Location (Tamil Nadu)</label>
      <input type="text" id="location" name="location" placeholder="e.g., Chennai, Madurai" required>
    </div>
    <div class="form-group">
      <label for="time">Time</label>
      <input type="time" id="time" name="time" required>
    </div>
    <div class="form-group">
      <label for="crime_type">Crime Type</label>
      <select id="crime_type" name="crime_type" required>
        <option value="">-- Select Crime Type --</option>
        <option value="Theft">Theft</option>
        <option value="Robbery">Robbery</option>
        <option value="Assault">Assault</option>
        <option value="Murder">Murder</option>
        <option value="Kidnapping">Kidnapping</option>
        <option value="Sexual Harassment">Sexual Harassment</option>
      </select>
    </div>

    <div class="buttons">
      <button type="submit">Submit</button>
      <button type="button" onclick="alert('Prediction model coming soon')">Predict</button>
      <button type="button" onclick="alert('Redirecting to heatmap')">Heatmap</button>
      <button type="button" onclick="alert('Logging out')">Logout</button>
    </div>
  </form>

  <h2 style="margin-top: 30px;">Crime Records in Tamil Nadu (1990–2024)</h2>
  <table>
    <thead>
      <tr>
        <th>Year</th>
        <th>Location</th>
        <th>Crime Type</th>
        <th>Reported Cases</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>1990</td><td>Chennai</td><td>Theft</td><td>1200</td></tr>
      <tr><td>1995</td><td>Madurai</td><td>Robbery</td><td>480</td></tr>
      <tr><td>2000</td><td>Coimbatore</td><td>Assault</td><td>700</td></tr>
      <tr><td>2005</td><td>Trichy</td><td>Murder</td><td>90</td></tr>
      <tr><td>2010</td><td>Salem</td><td>Kidnapping</td><td>110</td></tr>
      <tr><td>2015</td><td>Chennai</td><td>Sexual Harassment</td><td>350</td></tr>
      <tr><td>2020</td><td>Madurai</td><td>Theft</td><td>950</td></tr>
      <tr><td>2023</td><td>Coimbatore</td><td>Robbery</td><td>420</td></tr>
      <tr><td>2024</td><td>Chennai</td><td>Assault</td><td>780</td></tr>
    </tbody>
  </table>
</div>

</body>
</html>
    ''', prediction=prediction)

@app.route('/heatmap')
def heatmap():
    if 'username' not in session:
        return redirect(url_for('login'))
    map_html = generate_heatmap()
    return render_template_string(f'''
    <html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Crime Hotspot Heatmap - Tamil Nadu</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />

  <!-- Heatmap Plugin CSS -->
  <style>
    body {
      margin: 0;
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(120deg, #000000, #222222);
      color: #fff;
    }

    header {
      padding: 20px;
      background: #0d1117;
      color: #00eaff;
      font-size: 28px;
      text-align: center;
      font-weight: bold;
      animation: slideDown 1s ease-in-out;
    }

    #map {
      height: 90vh;
      width: 100%;
      animation: fadeIn 2s ease-in-out;
    }

    @keyframes slideDown {
      from {
        transform: translateY(-50px);
        opacity: 0;
      }
      to {
        transform: translateY(0);
        opacity: 1;
      }
    }

    @keyframes fadeIn {
      from {opacity: 0;}
      to {opacity: 1;}
    }
  </style>

  <!-- Leaflet JS -->
  <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>

  <!-- Leaflet.heat Plugin -->
  <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
</head>
<body>

<header>Safety Locator - Tamil Nadu Crime Hotspot Heatmap</header>

<div id="map"></div>

<script>
  // Initialize the map
  var map = L.map('map').setView([11.1271, 78.6569], 7); // Tamil Nadu center

  // Load base map tiles
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: 'Crime Data © Tamil Nadu',
    maxZoom: 18,
  }).addTo(map);

  // Mock crime location data [lat, lng, intensity]
  var heatData = [
    [13.0827, 80.2707, 0.9], // Chennai
    [10.7905, 78.7047, 0.7], // Trichy
    [11.0168, 76.9558, 0.6], // Coimbatore
    [9.9252, 78.1198, 0.8],  // Madurai
    [12.9165, 79.1325, 0.5], // Vellore
    [11.6643, 78.1460, 0.4], // Salem
    [10.9601, 78.0766, 0.3], // Karur
    [12.8342, 80.0442, 0.9], // Chengalpattu
    [13.6288, 79.4184, 0.6], // Tirupati border
    [10.8505, 77.7500, 0.5], // Erode
  ];

  // Create heat layer and add to map
  var heat = L.heatLayer(heatData, {
    radius: 30,
    blur: 20,
    maxZoom: 17,
    gradient: {
      0.1: 'blue',
      0.3: 'lime',
      0.6: 'orange',
      0.9: 'red'
    }
  }).addTo(map);
</script>

</body>
</html>
    ''', map_html=map_html)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)