import setuptools


setuptools.setup(
    name="numpygrad",
    version="0.1.0",
    author="Nguyen (Luca) Dang",
    author_email="nguyendanglmv@gmail.com",
    description="A tiny autograd engine wrote purely with numpy (contains, but not limited to, torch.nn, torch.nn.functional, torch.optim like functionalities)",
    long_description_content_type="text/markdown",
    url="https://github.com/nguyenddang/numpygrad",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)