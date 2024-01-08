![FaKe Logo](docs/fAke.png)

# FaKe : A Cross-Browser Extension For Classifying Filipino Fake News
Cross-browser extension for detecting Filipino fake news, powered by machine learning. 

## Description
This repository hosts a fake news classifier trained on a corpus of Filipino articles adapted from [Cruz et al](https://github.com/jcblaisecruz02/Tagalog-fake-news). As part of our ongoing undergraduate study at the University of the Philippines Visayas, we have built a cross-browser extension that employs our fake news classifier. We utilize [Tampermonkey](https://www.tampermonkey.net/) as a wrapper around our extension and [Render](https://render.com/) to host our machine learning model on the cloud as an API. The fake news classifier only works on articles written primarily in Filipino with smatterings of English vernacular. Currently, we have deployed a trained Logistic Regression model that has achieved a 97.4% accuracy on our test set. All resources that have been used in this study are available on this repository.

## Installation
1. Download and install [Tampermonkey](https://www.tampermonkey.net/) for your browser of choice.
2. Download the latest release of the extension from [releases](https://github.com/WhiteLicorice/Fake/releases).
3. Extract the zip file and open `script.js` with Notepad or your plaintext editor of choice.
4. Open the Tampermonkey dashboard.
5. Click the `+` button and it will open a new userscript.
6. Copy and paste the contents of `script.js` over the userscript, replacing the old contents.
7. Click `file` and then `save`.

## Usage
1. Pin Tampermonkey to your browser's address bar.
2. Once pinned, click the Tampermonkey icon.
3. Under `Fake News Detector`, click `Detect Fake News`.
4. Wait for the system to process the article you're reading.
5. Profit!

## Roles
1) `Rene Andre Jocsing`    Lead Programmer, Project Manager, Writer
2) `Chancy Ponce de Leon`  Programmer
3) `Cobe Austin Lupac`     Presenter, Programmer, Quality Assurance
4) `Ron Gerlan Naragdao`   Programmer

## Contributing
If you wish to contribute to the project, please open a pull request. Thank you.

## Resources
```
@inproceedings{cruz2020localization,
  title={Localization of Fake News Detection via Multitask Transfer Learning},
  author={Cruz, Jan Christian Blaise and Tan, Julianne Agatha and Cheng, Charibeth},
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
  pages={2596--2604},
  year={2020}
}

@article{evaluating2019cruz,
  title={{Evaluating Language Model Finetuning Techniques for Low-resource Languages}},
  author={Cruz, Jan Christian Blaise and Cheng, Charibeth},
  journal={arXiv preprint arXiv:1907.00409},
  year={2019}
}

@article{imperial2021application,
  author={Imperial, J. M. and Ong, E.},
  title={Application of Lexical Features Towards Improvement of Filipino Readability Identification of Children's Literature},
  journal={arXiv preprint},
  eprint={2101.10537},
  year={2021}
}

@article{imperial2021diverse,
  author={Imperial, J. M. and Ong, E.},
  title={Diverse Linguistic Features for Assessing Reading Difficulty of Educational Filipino Texts},
  journal={arXiv preprint},
  eprint={2108.00241},
  year={2021}
}

@inproceedings{imperial2020exploring,
  author={Imperial, J. M. and Ong, E.},
  title={Exploring Hybrid Linguistic Feature Sets To Measure Filipino Text Readability},
  booktitle={2020 International Conference on Asian Language Processing (IALP)},
  pages={175-180},
  year={2020},
  organization={IEEE}
}
```
