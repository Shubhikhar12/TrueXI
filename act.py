import streamlit_authenticator as stauth

# List of plaintext passwords
passwords = ['truexi123', 'admin456']

# Hash the passwords
hashed_passwords = stauth.utilities.hasher.Hasher(passwords).generate()

# Print the hashed list
print(hashed_passwords)
