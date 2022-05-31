from setuptools import setup, find_packages

setup(
    name='marley-chatbot',
    version='0.0.0',
    description='',
    url='https://github.com/orthose/marley-chatbot',
    author='Maxime VINCENT, Abderrahim BENMELOUKA, Loris PONROY, Lina SAICHI, Myriem MOULOUEL, Syrine MARZOUGUI',
    keywords='chatbot, airfrance, klm, flights',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['api-airfranceklm', 'nltk']
)