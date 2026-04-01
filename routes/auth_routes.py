from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models.user import User
import re

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
        
    if request.method == 'POST':
        action = request.form.get('action') # "login" or "register"
        
        if action == 'register':
            name = request.form.get('name')
            email = request.form.get('email')
            password = request.form.get('password')
            
            # Validation
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                flash('Invalid email address.', 'danger')
                return redirect(url_for('auth.login'))
                
            if User.get_by_email(email):
                flash('Email already registered!', 'danger')
                return redirect(url_for('auth.login'))
                
            User.create(name, email, password)
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('auth.login'))

        elif action == 'login':
            email = request.form.get('email')
            password = request.form.get('password')
            user = User.get_by_email(email)
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for('main.index'))
            else:
                flash('Login Unsuccessful. Please check email and password.', 'danger')

    return render_template('login.html')

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.first'))
