# shell.nix
{ pkgs ? import <nixpkgs> {} }:

with pkgs; mkShell rec {
  buildInputs = with python312Packages; [
    python312
    numpy
    networkx
    lxml
    tqdm
    typing-extensions
    soupsieve
    nltk
    joblib
    python312Packages.click
    beautifulsoup4
    scipy
    orjson
    (pkgs.python312Packages.buildPythonPackage rec {
      pname = "rocksdict";
      version = "0.3.25";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/ca/c9/355ec66af66f25d5130712f18d07ef1cd677582cd106438d5b1570db0b3b/rocksdict-0.3.25-cp313-cp313-manylinux_2_28_x86_64.whl";
        sha256 = "sha256-EBCdV1U41SerfeQxZoQAO2wpZo8DhyLiesrvZAccpIE="; # Replace with actual sha256
      };
      format = "wheel";
      doCheck = false;
    })
  ];
}
