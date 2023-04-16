from flask import Flask, render_template, request, redirect, url_for, flash, make_response, Response
from flask import Blueprint, render_template
from replit import Database
from argon2 import PasswordHasher
from passlib.hash import sha256_crypt
import cv2
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
import numpy as np
import pyaudio
import wave
import speech_recognition as sr
import openai


r = sr.Recognizer()
openai.api_key = "OPENAI-KEY" # Remove key in production

camera = cv2.VideoCapture(0)
ph = PasswordHasher()
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprised']

app = Flask(__name__)

db = Database(db_url="https://kv.replit.com/v0/eyJhbGciOiJIUzUxMiIsImlzcyI6ImNvbm1hbiIsImtpZCI6InByb2Q6MSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJjb25tYW4iLCJleHAiOjE2ODE2ODYxOTcsImlhdCI6MTY4MTU3NDU5NywiZGF0YWJhc2VfaWQiOiIxMWI5NGY4OC0yMWM3LTRiOGEtOTAxMC1jMDAwMzZkZTYwNzQiLCJ1c2VyIjoiQzBEM1cxWiIsInNsdWciOiJTQUFJLURCIn0.ki4Wb6MASz0X7NkkFc0NGnMMmNVUhOKtjRGWpTGsq_J_gOob7RUpIxv9Y9eGD8_X8ZWcpqvCCuJGVMnRDEltyQ")

@app.route("/")
def welcome():
  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  perms = 'none'
  if loggedIn == "true":
    if username != None and username in db.keys():
      if ph.verify(session, username) == True:
        perms = db[username+"stat"]
  return render_template("index.html", loggedIn=loggedIn, perms = perms)

@app.route("/panel", methods=["GET", "POST"])
def panel():

  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  perms = 'none'
  if loggedIn == "true":
    if username != None and username in db.keys():
      if ph.verify(session, username) == True:
        perms = db[username+"stat"]
  
  if request.method == "POST":    
    currpass = request.form.get("currpass")
    newpass = request.form.get("newpass")
    repass = request.form.get("repass")
    
    if sha256_crypt.verify(currpass, db[username]) == True:
      if newpass == repass:
        db[username] = sha256_crypt.encrypt(newpass)
        print(db[username])

  if loggedIn == "true":
    if username != None and username in db.keys():
      if ph.verify(session, username) == True:
        return render_template("panel.html", loggedIn = loggedIn, username = username, session = session, perms = perms)
    else:
      return redirect("/logout")
  else:
    return redirect("/login")


@app.route("/chat", methods=["GET", "POST"])
def chat():
  text2 = "none"
  s = "none"
  text = "none"
  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")

  if request.method == 'POST':
            
    freq = 44100

    # Recording duration
    duration = 10
    
    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                      samplerate=freq, channels=2)
    
    # Record audio for the given number of seconds
    sd.wait()
    
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("recording0.wav", freq, recording)
    
    # Convert the NumPy array to audio file
    wv.write("recording1.wav", recording, freq, sampwidth=2)

    srtran = sr.AudioFile('recording1.wav')
    with srtran as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        print("Text: "+s)
    except Exception as e:
        print("Exception: "+str(e))
    
    completion = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"What kind of emotion is this text expressing, say it in no more than one word from these emotions (Happy, Angry, Surprise, Sad, Fear, and Neutral): {s}\nAI:",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7, 
            )

    for choice in completion.choices:
        language = 'en'
        text=choice.text.strip()

    completion2 = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Using this input from the user: {s}, and the emotion of the user from that text: {text}, say something which will help their mode and give recommondations to make their situation better.",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7, 
            )

    for choice2 in completion2.choices:
        language = 'en'
        text2=choice2.text.strip()
        print(text2)



  perms = 'none'
  if loggedIn == "true":
    if username != None and username in db.keys():
      if ph.verify(session, username) == True:
        perms = db[username+"stat"]
  return render_template("chat.html", loggedIn=loggedIn, perms = perms, text2=text2, ftext=s, text=text)


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/settings", methods=["GET", "POST"])
def settings():

  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  perms = 'none'
  if loggedIn == "true":
    if username != None and username in db.keys():
      if ph.verify(session, username) == True:
        perms = db[username+"stat"]
  
  if request.method == "POST":    
    currpass = request.form.get("currpass")
    newpass = request.form.get("newpass")
    repass = request.form.get("repass")
    
    if sha256_crypt.verify(currpass, db[username]) == True:
      if newpass == repass:
        db[username] = sha256_crypt.encrypt(newpass)
        print(db[username])

  if loggedIn == "true":
    if username != None and username in db.keys():
      if ph.verify(session, username) == True:
        return render_template("settings.html", loggedIn = loggedIn, username = username, session = session, perms = perms)
    else:
      return redirect("/logout")
  else:
    return redirect("/login")

@app.route('/')
def index():
    return render_template('panel.html')


def clear():
  for i in db.keys():
    del db[i]

@app.route("/login")
def login():
  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  perms = 'none'
  if loggedIn == "true":
    if username != None and username in db.keys():
      if ph.verify(session, username) == True:
        perms = db[username+"stat"]
  if loggedIn == "true":
    return redirect("/home")
  else:
    return render_template("login.html", loggedIn = loggedIn, username = username, session = session, perms = perms)

@app.route("/signup")
def signup():
  loggedIn = request.cookies.get("loggedIn")
  username = request.cookies.get("username")
  session = request.cookies.get("session")
  perms = 'none'
  if loggedIn == "true":
    if username != None and username in db.keys():
      if ph.verify(session, username) == True:
        perms = db[username+"stat"]
  if loggedIn == "true":
    return redirect("/home")
  else:
    return render_template("signup.html", loggedIn = loggedIn, username = username, session = session, perms = perms)

@app.route("/loginsubmit", methods=["GET", "POST"])
def loginsubmit():
  if request.method == "POST":
    username = request.form.get("username")
    password = request.form.get("password")
    if username in db.keys():
      if sha256_crypt.verify(password, db[username]) == True:
        resp = make_response(render_template('readcookie.html'))
        resp.set_cookie("loggedIn", "true")
        resp.set_cookie("username", username)
        resp.set_cookie("session", ph.hash(username))
        return resp
      else:
        return render_template("error.html", error="Incorrect Password, please try again.")
    else:
      return render_template("error.html", error="Account does not exist, please sign up.")

@app.route("/createaccount", methods=["GET", "POST"])
def createaccount():
  if request.method == "POST":
    newusername = request.form.get("newusername")
    newpassword = sha256_crypt.encrypt((request.form.get("newpassword")))
    orignewpass = request.form.get("newpassword")
    reenterpassword = request.form.get("reenterpassword")
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    cap_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    allchars = letters + cap_letters + numbers + ['_']
    print(newusername)
    for i in newusername:
      if i not in allchars:
        return "Username can only contain alphanumeric characters and underscores."
    if newusername in db.keys():
      return render_template("error.html", error="Username taken.")
    if newusername == "":
      return render_template("error.html", error="Please enter a username.")
    if newpassword == "":
      return render_template("error.html", error="Please enter a password.")
    if reenterpassword == orignewpass:
      db[newusername] = newpassword
      db[newusername+"stat"] = "user"
      resp = make_response(render_template('readcookie.html'))
      resp.set_cookie("loggedIn", "true")
      resp.set_cookie("username", newusername)
      resp.set_cookie("session", ph.hash(newusername))
      return resp
    else:
      return render_template("error.html", error="Passwords don't match.")

@app.route("/logout")
def logout():
  resp = make_response(render_template('readcookie.html'))
  resp.set_cookie("loggedIn", "false")
  resp.set_cookie("username", "None")
  return resp


def get_frame():
    if camera.isOpened():
        ret, frame = camera.read() 
        if ret:
            return(ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return ret, None
    else:
        return None
    


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the cap frame
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),10)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)


                    preds = classifier.predict(roi)[0]
                    #print("\nprediction = ",preds)
                    label=class_labels[preds.argmax()]
                    print(label)

                    
                    #print("\nprediction max = ",preds.argmax())
                    #print("\nlabel = ",label)
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(40, 224, 129),5)
                else:
                    cv2.putText(frame,'Please make certain there is a face in front of the Camera.',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(40, 224, 129),3)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result




if __name__ == "__main__":
  app.run(debug=True, port=5000, host='0.0.0.0')

#======================================================================================================#   


# with sr.Microphone() as source:
#     r.adjust_for_ambient_noise(source)
#     audio = r.listen(source)

#     try:
#         text = r.recognize_google(audio)

#         completion = openai.Completion.create(
#                 engine="text-davinci-002",
#                 prompt=f"What kind of emotion is this text expressing, say it in no more than one word from these emotions (Happy, Angry, Surprise, Sad, Fear, and Neutral): {text}\nAI:",
#                 max_tokens=1024,
#                 n=1,
#                 stop=None,
#                 temperature=0.7, 
#                 )

#         for choice in completion.choices:
#             language = 'en'
#             text=choice.text.strip()
#             print(text)
        
#     except sr.UnknownValueError:
#         print("Could not understand audio")
#     except sr.RequestError as e:
#         print("Could not request results from Google Speech Recognition service; {0}".format(e))

#======================================================================================================#    



# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier = load_model('Emotion_Detection.h5')

# class_labels = ['Angry','Happy','Neutral','Sad','Surprised']




# while True:
#     # Grab a single frame of video
#     ret, frame = camera.read()
#     labels = []
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray,1.3,5)

#     for (x,y,w,h) in faces:
#         # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),10)
#         roi_gray = gray[y:y+h,x:x+w]
#         roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


#         if np.sum([roi_gray])!=0:
#             roi = roi_gray.astype('float')/255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi,axis=0)


#             preds = classifier.predict(roi)[0]
#             #print("\nprediction = ",preds)
#             label=class_labels[preds.argmax()]
#             print(label)
#             #print("\nprediction max = ",preds.argmax())
#             #print("\nlabel = ",label)
#             # label_position = (x,y)
#             # cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(40, 224, 129),5)
#         else:
#             print("No face detected")
#             # cv2.putText(frame,'Please make certain there is a face in front of the Camera.',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(40, 224, 129),3)
#     # cv2.imshow('Emotion Detector',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # camera.release()
# # cv2.destroyAllWindows()
