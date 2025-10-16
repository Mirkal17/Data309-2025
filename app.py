from agents import *
from flask import Flask, render_template, redirect, request
app = Flask(__name__)

messages = [] 
@app.route('/', methods=['GET', 'POST']) 
def home(): 
    if request.method == 'POST': 
        user_message = request.form.get('message') 
        if user_message: 
            messages.append({ 
            'user':user_message, 
            'bot': evaluate_ticket(user_message) }) 
    return render_template('index.html', messages=messages)


@app.route('/clear', methods=['POST'])
def clear_chat():
    global messages
    messages = []  # reset chat history
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)