import click
from src.predict_model import save_preds, evaluate_preds

@click.command()
@click.option('--save', is_flag = True, 
              help='load a trained learner, make predictions and save to df locally')
@click.option('--eval', is_flag=True, 
              help='evaluate predictions')

def main(save, eval):
    if save:
        save_preds()
    elif eval:
        evaluate_preds()
    else:
        click.echo("Please specify a mode (--help for more info)")
        
if __name__=="__main__":
    main()