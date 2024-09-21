
import click
import subprocess
import os
import json
import time
import random
import configparser
import itertools
# Define the scripts and their paths
# Define the scripts and their paths
SCRIPTS = [
    ("Fetch Data", "D:\\code\\trading_system\\scripts\\fetch_data.py", "Fetch the latest market data."),
    ("Add Indicators", "D:\\code\\trading_system\\scripts\\add_indicators.py", "Add technical indicators to the data."),
    ("Train Model", "D:\\code\\trading_system\\scripts\\train_save_model.py", "Train the machine learning model."),
    ("Backtest", "D:\\code\\trading_system\\scripts\\backtest.py", "Run a backtest on historical data."),
    ("Calculate Portfolio Size", "D:\\code\\trading_system\\scripts\\portfolio_size.py", "Calculate the optimal portfolio size."),
    ("Process Data", "D:\\code\\trading_system\\scripts\\process_data.py", "Process raw data for analysis."),
    ("Generate Signals", "D:\\code\\trading_system\\scripts\\generate_signals.py", "Generate trading signals."),
    ("Make Predictions", "D:\\code\\trading_system\\scripts\\make_predictions.py", "Make predictions using the trained model."),
]


SEQUENCES_FILE = 'sequences.json'
LOG_FILE = 'execution_log.txt'
CONFIG_FILE = 'D:/code/trading_system/data/config.ini'

QUOTES = [
    "I don't predict market movements. I create them.",
    "Risk isn't uncertainty. Risk is not knowing how much uncertainty you're dealing with.",
    "The market is my canvas, volatility my paint, and profits my masterpiece.",
    "I don't follow trends. I set them, then profit from those who follow.",
    "My strategy? Simple. Be fearful when others are greedy, and greedy when others are fearful... on steroids.",
    "I don't time the market. I make the market time me.",
    "Wealth isn't about having money, it's about having options. And I trade both.",
    "The best traders don't chase the news. We create it.",
    "My edge isn't information. It's how fast I process it and act on it.",
    "I don't diversify to reduce risk. I concentrate to magnify returns.",
    "The market doesn't beat me. I am the market.",
    "Others see patterns in the noise. I see opportunities in the chaos.",
    "I'm not lucky. I'm just prepared for scenarios others haven't imagined yet.",
    "My biggest positions? Those are just warm-ups.",
    "I don't react to the market. The market reacts to me.",
    "Losing trades? You mean data points for future victories.",
    "The difference between gambling and trading? I control the odds.",
    "I'm not a day trader. I'm a 'seize the day' trader.",
    "Market crashes don't scare me. They're just flash sales for the prepared.",
    "Why fear volatility when you can harness it?",
    "I don't buy the dip. I create the dip, then buy the recovery.",
    "My portfolio isn't diverse. It's precisely calibrated for maximum impact.",
    "The market is efficient? Please. It's only as efficient as its least efficient player.",
    "I don't have inside information. I have outside observation skills.",
    "Bulls make money, bears make money, but I make money in any zoo.",
    "The trend is your friend? No, the trend is my employee.",
    "I don't use stop losses. I use strategic retreats for bigger advances.",
    "Market sentiment? That's just another tradable asset.",
    "I don't trade the market. I trade human psychology.",
    "Fundamentals are nice, but I trade the future, not the past.",
    "Technical analysis isn't about predicting. It's about probability management.",
    "I'm not overleveraged. I'm optimally positioned for exponential returns.",
    "Risk management isn't about avoiding risks. It's about embracing the right ones.",
    "The market can stay irrational longer than you can stay solvent? Watch me outlast it.",
    "I don't need algorithms. I am the algorithm.",
    "Shorting isn't pessimism. It's optimism about the market's inefficiencies.",
    "My best trade? The one I haven't made yet.",
    "I don't follow smart money. I lead it.",
    "Market cycles? I prefer to call them profit oscillations.",
    "The invisible hand of the market? I'm the one shaking it.",
    "I don't invest in companies. I invest in paradigm shifts.",
    "Patience isn't waiting for opportunities. It's setting traps for them.",
    "The market doesn't reward diversification. It rewards calculated concentration.",
    "I'm not a contrarian. I'm a step ahead of the consensus.",
    "My risk tolerance isn't high. My risk understanding is.",
    "I don't time bottoms. I create them for others.",
    "Market efficiency is just an opportunity in disguise.",
    "I don't trade assets. I trade market psychology.",
    "The best hedge against market uncertainty? Superior information processing.",
    "I'm not beating the market. I'm redefining what 'market' means.",
    "Liquidity crisis? You mean buying opportunity.",
    "I don't need to be right all the time. I just need to be right when it matters most.",
    "Market manipulation? I prefer the term 'strategic influence'.",
    "I don't ride waves. I create tsunamis.",
    "The market is a voting machine in the short term, a weighing machine in the long term, and my playground always.",
    "I don't predict black swan events. I profit from the market's overreaction to them.",
    "Market bubbles aren't dangers. They're profit balloons waiting to be popped.",
    "I don't fear market corrections. I orchestrate them.",
    "The efficient market hypothesis? A theory for those who can't see the inefficiencies.",
    "I'm not a speculator. I'm an opportunity capitalizer.",
    "Risk-free rate? In my world, even cash is a position.",
    "I don't need diversification. I need precision in execution.",
    "Market volatility isn't noise. It's the music I dance to.",
    "I don't compete with other traders. I compete with market inefficiencies.",
    "The market isn't always right, but it's always a source of opportunity.",
    "I don't follow economic indicators. I trade the reactions to them.",
    "Beating the market isn't hard. Beating yourself is the real challenge.",
    "I don't need a crystal ball. I have real-time data and superior analysis.",
    "The market is my opponent, and I always play to win.",
    "I don't avoid drawdowns. I use them as springboards.",
    "Market panic? That's just adrenaline for my trades.",
    "I'm not a market timer. I'm a market rhythm master.",
    "The best trade is often the one you don't make, but the most profitable is the one only you see.",
    "I don't fight the Fed. I anticipate its moves and profit accordingly.",
    "Market anomalies aren't errors. They're invitations for profit.",
    "I don't need luck. I create my own probability fields.",
    "The market isn't a zero-sum game. It's a game of skill, and I'm a grandmaster.",
    "I don't follow market leaders. I lead market followers.",
    "Risk isn't my enemy. Missed opportunity is.",
    "I don't trade the news. I trade the market's misinterpretation of it.",
    "Market cycles are just my profit cycles in disguise.",
    "I don't need market stability. I thrive in chaos.",
    "The market doesn't give second chances, but it does give new opportunities.",
    "I'm not afraid of leverage. I'm afraid of not using it properly.",
    "Market sentiment is just another tradable commodity.",
    "I don't wait for the perfect setup. I create it.",
    "The market is never wrong, but it is often mispriced.",
    "I don't need to know what's going to happen. I just need to know how to react.",
    "Market consensus is just another contrarian indicator.",
    "I don't trade markets. I trade expectations.",
    "The best traders don't have the most information. They have the best information filters.",
    "I'm not in the prediction business. I'm in the probability business.",
    "Market fear is just opportunity wearing a mask.",
    "I don't follow the smart money. I anticipate where it's going.",
    "The market is my employee, not my boss.",
    "I don't need to be popular. I need to be profitable.",
    "Market timing isn't about being right. It's about being right enough.",
    "I don't chase returns. I create them.",
    "The market is a game of chess, and I'm always several moves ahead.",
    "I don't react to the market. I make the market react to me."
]

ASCII_ART = r"""
 _____                                                             _____ 
( ___ )                                                           ( ___ )
 |   |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|   | 
 |   |   /$$$$$$  /$$$$$$$$ /$$$$$$  /$$   /$$ /$$   /$$  /$$$$$$  |   | 
 |   |  /$$__  $$|__  $$__//$$__  $$| $$$ | $$| $$  /$$/ /$$__  $$ |   | 
 |   | | $$  \__/   | $$  | $$  \ $$| $$$$| $$| $$ /$$/ | $$  \__/ |   | 
 |   | |  $$$$$$    | $$  | $$  | $$| $$ $$ $$| $$$$$/  |  $$$$$$  |   | 
 |   |  \____  $$   | $$  | $$  | $$| $$  $$$$| $$  $$   \____  $$ |   | 
 |   |  /$$  \ $$   | $$  | $$  | $$| $$\  $$$| $$\  $$  /$$  \ $$ |   | 
 |   | |  $$$$$$/   | $$  |  $$$$$$/| $$ \  $$| $$ \  $$|  $$$$$$/ |   | 
 |   |  \______/    |__/   \______/ |__/  \__/|__/  \__/ \______/  |   | 
 |___|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|___| 
(_____)                                                           (_____)
"""

def load_sequences():
    if os.path.exists(SEQUENCES_FILE):
        with open(SEQUENCES_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_sequences(sequences):
    with open(SEQUENCES_FILE, 'w') as file:
        json.dump(sequences, file, indent=4)

def log_execution(script_name, duration, success):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {script_name} - Duration: {duration:.2f}s - {'Success' if success else 'Failed'}\n")

def run_script(script_path):
    try:
        start_time = time.time()
        subprocess.run(['python', script_path], check=True)
        duration = time.time() - start_time
        log_execution(script_path, duration, True)
        click.secho(f"Execution time: {duration:.2f} seconds", fg="green")
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        log_execution(script_path, duration, False)
        click.secho(f"Error while running {script_path}: {e}", fg="red")

def update_config(params):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    for section, parameters in params.items():
        if section not in config:
            config.add_section(section)
        for key, value in parameters.items():
            config.set(section, key, str(value))
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

def get_config_sections():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config.sections()

def get_config_keys(section):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config.options(section)

def show_config_sections():
    sections = get_config_sections()
    click.echo(click.style("Available Config Sections:", fg="blue"))
    for section in sections:
        click.echo(click.style(f"[{section}]", fg="cyan"))

def show_config_keys(section):
    keys = get_config_keys(section)
    click.echo(click.style(f"Available Keys in Section [{section}]:", fg="blue"))
    for key in keys:
        click.echo(click.style(f"  {key}", fg="green"))

@click.group()
def cli():
    pass

@click.command()
def dashboard():
    """Interactive dashboard to list, run, and save scripts."""
    while True:
        click.clear()
        click.echo(click.style(ASCII_ART, fg="yellow"))
        click.echo(click.style("Welcome to the Trading Dashboard!", fg="cyan", bold=True))
        click.echo(click.style(random.choice(QUOTES), fg="magenta", italic=True))
        
        click.echo("\nAvailable scripts:")
        for idx, (name, _, description) in enumerate(SCRIPTS, start=1):
            click.echo(click.style(f"{idx}. {name} - {description}", fg="cyan"))
        
        click.echo("\nOptions:")
        click.echo(click.style("1. Run scripts by index", fg="cyan"))
        click.echo(click.style("2. Save a sequence of scripts", fg="cyan"))
        click.echo(click.style("3. Run a saved sequence", fg="cyan"))
        click.echo(click.style("4. List saved sequences", fg="cyan"))
        click.echo(click.style("5. Run script(s) with different parameters", fg="cyan"))
        click.echo(click.style("6. Exit", fg="cyan"))
        
        choice = click.prompt(click.style("Choose an option", fg="green", bold=True), type=int)

        if choice == 1:
            indices = click.prompt(click.style("Enter script indices separated by dots (e.g., 1.2.3)", fg="green"))
            indices = [int(idx) - 1 for idx in indices.split('.')]
            for idx in indices:
                if 0 <= idx < len(SCRIPTS):
                    name, path, _ = SCRIPTS[idx]
                    click.echo(click.style(f"Running {name}...", fg="yellow"))
                    run_script(path)
                else:
                    click.echo(click.style(f"Invalid index: {idx + 1}", fg="red"))
            click.pause()
        
        elif choice == 2:
            name = click.prompt(click.style("Enter a name for the sequence", fg="green"))
            indices = click.prompt(click.style("Enter script indices separated by dots (e.g., 1.2.3)", fg="green"))
            sequences = load_sequences()
            sequences[name] = indices
            save_sequences(sequences)
            click.echo(click.style(f"Sequence '{name}' saved.", fg="green"))
            click.pause()

        elif choice == 3:
            sequences = load_sequences()
            if not sequences:
                click.echo(click.style("No sequences found.", fg="red"))
            else:
                click.echo(click.style("Available sequences:", fg="blue"))
                for name in sequences:
                    click.echo(click.style(f"- {name}", fg="blue"))
                name = click.prompt(click.style("Enter the name of the sequence to run", fg="green"))
                if name in sequences:
                    indices = sequences[name]
                    indices = [int(idx) - 1 for idx in indices.split('.')]
                    for idx in indices:
                        if 0 <= idx < len(SCRIPTS):
                            name, path, _ = SCRIPTS[idx]
                            click.echo(click.style(f"Running {name}...", fg="yellow"))
                            run_script(path)
                        else:
                            click.echo(click.style(f"Invalid index: {idx + 1}", fg="red"))
                else:
                    click.echo(click.style("Sequence '{name}' not found.", fg="red"))
            click.pause()

        elif choice == 4:
            sequences = load_sequences()
            if not sequences:
                click.echo(click.style("No sequences found.", fg="red"))
            else:
                click.echo(click.style("Saved sequences:", fg="blue"))
                for name, indices in sequences.items():
                    click.echo(click.style(f"{name}: {indices}", fg="blue"))
            click.pause()

        elif choice == 5:
            script_indices = click.prompt(click.style("Enter the script indices to run with different parameters, separated by dots (e.g., 1.2.3)", fg="green"))
            indices = [int(idx) - 1 for idx in script_indices.split('.')]
            param_list = []
            while True:
                show_config_sections()
                section = click.prompt(click.style("Enter the config section", fg="green"))
                show_config_keys(section)
                key = click.prompt(click.style("Enter the parameter key", fg="green"))
                values = click.prompt(click.style("Enter the parameter values separated by commas (e.g., 1,2,3)", fg="green"))
                values = values.split(',')
                param_list.append((section, key, values))
                if click.confirm(click.style("Do you want to add another parameter?", fg="green"), default=False):
                    continue
                else:
                    break
            
            param_combinations = list(itertools.product(*[[(section, key, value) for value in values] for section, key, values in param_list]))

            for idx in indices:
                if 0 <= idx < len(SCRIPTS):
                    script_name, script_path, _ = SCRIPTS[idx]
                    click.echo(click.style(f"Running {script_name} with different parameters...", fg="yellow"))

                    for param_set in param_combinations:
                        config_update = {}
                        for section, key, value in param_set:
                            if section not in config_update:
                                config_update[section] = {}
                            config_update[section][key] = value
                        click.echo(click.style(f"Updating config with: {config_update}", fg="cyan"))
                        update_config(config_update)
                        run_script(script_path)
                else:
                    click.echo(click.style(f"Invalid index: {idx + 1}", fg="red"))
            click.pause()

        elif choice == 6:
            click.echo(click.style("Exiting the dashboard. See you next time!", fg="cyan", bold=True))
            break

        else:
            click.echo(click.style("Invalid option. Please choose again.", fg="red"))

cli.add_command(dashboard)

if __name__ == "__main__":
    cli()
