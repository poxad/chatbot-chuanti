import streamlit_authenticator as stauth

# List of plaintext passwords
passwords = ['abc', 'def','pox']

# Hash the passwords
hashed_passwords = stauth.Hasher(passwords).generate()

# Print the hashed passwords to replace in your YAML file
print(hashed_passwords)
