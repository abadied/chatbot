#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script for safety protected interaction between a local human keyboard input and a
trained model.
"""

from parlai.core.params import ParlaiParser
from parlai.scripts.script import ParlaiScript
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.safe_local_human.safe_local_human import SafeLocalHumanAgent
import parlai.utils.logging as logging
import random
import os
from parlai.core.opt import Opt
from flask import Flask, render_template
from flask import request

application = Flask(__name__)

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Interactive chat with a model')
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument(
        '--display-prettify',
        type='bool',
        default=False,
        help='Set to use a prettytable when displaying '
        'examples with text candidates',
    )
    parser.add_argument(
        '--display-ignore-fields',
        type=str,
        default='label_candidates,text_candidates',
        help='Do not display these fields',
    )
    parser.add_argument(
        '-it',
        '--interactive-task',
        type='bool',
        default=True,
        help='Create interactive version of task',
    )
    parser.set_defaults(interactive_mode=True, task='interactive')
    SafeLocalHumanAgent.add_cmdline_args(parser)
    return parser


world = None
agent = None
opt = None

@application.route('/', methods=['GET', 'POST'])
def safe_interactive():
    global world, agent, opt
    random.seed(42)
    if request.method == 'GET':
        parser = ParlaiParser(True, True, 'Interactive chat with a model')
        parser.add_argument('-d', '--display-examples', type='bool', default=False)
        parser.add_argument(
            '--display-prettify',
            type='bool',
            default=False,
            help='Set to use a prettytable when displaying '
                 'examples with text candidates',
        )
        parser.add_argument(
            '--display-ignore-fields',
            type=str,
            default='label_candidates,text_candidates',
            help='Do not display these fields',
        )
        parser.add_argument(
            '-it',
            '--interactive-task',
            type='bool',
            default=True,
            help='Create interactive version of task',
        )
        parser.set_defaults(interactive_mode=True, task='interactive')
        SafeLocalHumanAgent.add_cmdline_args(parser)
        opt = parser.parse_args(args=None, print_args=False)

        if parser is not None:
            if parser is True and isinstance(opt, ParlaiParser):
                parser = opt
            elif parser is False:
                parser = None
        if isinstance(opt, ParlaiParser):
            logging.error('interactive should be passed opt not Parser')
            opt = opt.parse_args()

        opt['model_file'] = os.getcwd() + '\\data\models\\blender/blender_90M/model'

        # Create model and assign it to the specified task
        agent = create_agent(opt, requireModelExists=True)
        if parser:
            # Show arguments after loading model
            parser.opt = agent.opt
            parser.print_args()
        human_agent = SafeLocalHumanAgent(opt)
        world = create_task(opt, [human_agent, agent])
        return render_template('chatbot.html')

    # Interact until episode done
    if request.method == 'POST':
        sentence = request.form['msg']
        world.parley(reply_text=sentence)
        bot_act = world.get_acts()[-1]
        if 'bot_offensive' in bot_act and bot_act['bot_offensive']:
            agent.reset()

        if opt.get('display_examples'):
            print('---')
            print(world.display())
        if world.epoch_done():
            logging.info('epoch done')
        return bot_act['text']


if __name__ == '__main__':
    application.debug = True
    application.run()



