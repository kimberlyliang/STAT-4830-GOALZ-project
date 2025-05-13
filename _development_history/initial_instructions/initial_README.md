# STAT 4830 Project Repository

Welcome to your project repository! This template helps you develop and implement an optimization project over the semester.

## Getting Started

1. **Finding Your Project Idea**
   - Start with our [Project Ideas Guide](docs/finding_project_ideas.md)
   - Use AI to explore and refine your ideas
   - Take time to find something you care about

   It's very important you learn to use AI tools in your work! [Noam Brown](https://x.com/polynoamial/status/1870307185961386366) (OpenAI) says that students should...
   > Practice working with AI. Human+AI will be superior to human or AI alone for the foreseeable future. Those who can work most effectively with AI will be the most highly valued.

   ![Noam tweet](figures/noam.png)

2. **Week 3 Deliverable**
   - Follow the [Week 3 Instructions](docs/assignments/week3_deliverable_instructions.md)
   - Required components:
     - Initial report draft
     - Self-critique document analyzing your report's strengths and weaknesses
     - Supporting Jupyter notebooks/code
   - Due: Friday, January 31, 2025

## Project Development Cycle

Each week follows an OODA (Observe, Orient, Decide, Act) loop that helps you improve your project systematically:

![Project Development Cycle - A diagram showing the OODA loop (Observe, Orient, Decide, Act) adapted for project development. Each phase has specific activities: Observe (Review Report, Check Results), Orient (Write Critique, Find Gaps), Decide (Plan Changes, Set Goals), and Act (Code, Run Tests). The phases are connected by arrows showing the flow of work, with a feedback loop labeled "Iterative Development" completing the cycle.](docs/figures/ooda_loop.png)

Each cycle produces specific deliverables:
- OBSERVE: Updated report draft
- ORIENT: Self-critique document
- DECIDE: Next actions plan
- ACT: Code changes & results

See the [Week 3 Instructions](docs/assignments/week3_deliverable_instructions.md) for detailed guidance on writing your first self-critique.

## Project Schedule

### Deliverables (Due Fridays)
- Week 2 (Jan 24): Email Project Team Names to yihuihe@wharton.upenn.edu
- Week 3 (Jan 31): Report Draft 1 + Code + Self Critique
- Week 4 (Feb 7): Slides Draft 1
- Week 5 (Feb 14): Report Draft 2 + Code + Self Critique
- Week 6 (Feb 21): Slides Draft 2
- Week 7 (Feb 28): Report Draft 3 + Code + Self Critique
- Week 8: ⚡ Lightning Talks in Class (Mar 5/7) & Slides Draft 3 due Friday ⚡
- Spring Break (Mar 8-16)
- Week 9 (Mar 21): Report Draft 4 + Code + Self Critique
- Week 10 (Mar 28): Slides Draft 4
- Week 11 (Apr 4): Report Draft 5 + Code + Self Critique
- Week 12 (Apr 11): Slides Draft 5
- Week 13: Final Presentations in Class (Apr 22/24) & Report Draft 6 + Code + Self Critique due Friday (Apr 18)
- Week 14 (Apr 29): Final Report + Code + Self Critique Due

Note: Instructions for peer feedback will be added throughout the semester for each deliverable.

Each draft builds on the previous one, incorporating feedback and new results. You'll meet with course staff three times during the semester to discuss your progress.

## Project Grading

Each deliverable is graded on five components:
- Report (20%): Problem statement, methodology, results
- Implementation (35%): Working code, tests, experiments
- Development Process (15%): Logs, decisions, iterations
- Critiques (15%): Reflection and planning
  - Self-critiques (required)
  - Peer critiques (when assigned)
  - Response to feedback
- Repository Structure (15%): Organization, documentation, clarity

Remember:
- Quality > Quantity
- Working > Perfect

## Repository Structure

```
your-repo/
├── README.md                    # This file
├── report.md                    # Your project report
├── notebooks/                   # Jupyter notebooks
├── src/                        # Source code
├── tests/                      # Test files
└── docs/
    ├── finding_project_ideas.md    # Guide to finding your project
    ├── assignments/                # Assignment instructions
    ├── llm_exploration/           # AI conversation logs
    └── development_log.md         # Progress & decisions
```

## Development Environment

### Editor Setup
We recommend using one of these editors:

1. **VS Code** (Free, Industry Standard)
   - Download from https://code.visualstudio.com/
   - Install recommended extensions:
     - Python
     - GitHub Pull Requests
     - GitHub Copilot (FREE for students!)
       - Sign up at https://education.github.com/discount_requests/application
       - This gives you FREE access to GitHub Copilot
       - Plus other GitHub student benefits

2. **Cursor** (Paid Alternative, $20/month)
   - Built on VS Code with additional AI features
   - Same interface and shortcuts as VS Code
   - Same extensions work
   - Added AI assistance for code exploration

Both editors work well with Git and provide excellent AI assistance. VS Code with Copilot is recommended for beginners as it's free with your student status and is the industry standard.

### Required Tools
- Python 3.10+
- PyTorch
- Jupyter Notebook/Lab
- Git

## Git Setup and Workflow

### First Time Setup
1. Fork this repository
   - Click "Fork" in the top right
   - Name it `STAT-4830-[team-name]-project`
   - This creates your own copy that can receive updates

2. Set up Git (if you haven't already):
   If you installed VS Code or Cursor, they'll help you install Git! Both editors have excellent Git integration built in.
   
   For detailed instructions, see the [Official Git installation guide](https://github.com/git-guides/install-git)

   After installing, set up your identity:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@upenn.edu"
   ```

3. Clone your fork:
   ```bash
   # HTTPS (easier):
   git clone https://github.com/[your-username]/STAT-4830-[team-name]-project.git

   # SSH (if you've set up SSH keys):
   git clone git@github.com:[your-username]/STAT-4830-[team-name]-project.git
   
   cd STAT-4830-[team-name]-project
   ```

4. Add upstream remote (to get updates):
   ```bash
   # HTTPS:
   git remote add upstream https://github.com/damek/STAT-4830-project-base.git

   # SSH:
   git remote add upstream git@github.com:damek/STAT-4830-project-base.git
   ```

5. Add your team members as collaborators:
   - Go to your repo on GitHub
   - Settings → Collaborators → Add people
   - Add using their GitHub usernames

### Working on Your Project
1. Create a new branch:
   ```bash
   git checkout -b exploration
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin exploration
   ```

### Getting Updates
When the base repository is improved:
```bash
# Get updates
git fetch upstream
git checkout main
git merge upstream/main

# Update your branch
git checkout exploration
git merge main
```

### Troubleshooting
- Having Git issues? Post on Ed Discussion
- Can't push/pull? Check if you're using HTTPS or SSH
- Windows path too long? Enable long paths:
  ```bash
  git config --system core.longpaths true
  ```

## Getting Help
- Use AI tools (ChatGPT, GitHub Copilot)
- See course staff for technical issues
- Document your progress






