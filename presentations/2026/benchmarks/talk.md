# Benchmarks are multipurpose. 

I'm going to make the bull case for benchmarks, and this talk will focus on public benchmarks and why they are useful.

Audiences. 

The most common usage is to assess whether or not a model is good at a type of task, or better than another model. Like any test with a score they are competitive.

For harness designers and integrators, recent price drops have made benchmarking as integration/optimisation viable. Does this skill or system prompt change improve accuracy, or save tokens? 

Quality Assurance. Is the model configured correctly? Does the harness work efficiently? 

Does your setup approach the number reported in the model card? 


## Before we Continue

This presentation is about openness. Themes discussed here are possible because of sharing data, reviewing and making evidence available. 



## Accessible

Considerable planning would go in to running, costs were high. 




## Getting a Score

Comparable - like the range or mileage on car. If the manufacturer makes a claim, I want to verify that claim.

## No refunds for bad tokens

If the vendor reports a score/cost on their model card, then you know that 

The nice thing about inference business


# What's behind the number?

To actually calculate a number, a few things needs to be in place. The model itself, and its hosting environment, then a way of interacting with the model: usually a harness running in an environment. And we need a way of giving the model some work to do - through the harness - and then assessing it's success and cost.

One of the most common benchmarks in recent times is terminal-bench 2.1. It's an open source benchmark, and contains 89 trials. Each trial is repeated 5 times.  In a typical run easy tasks pass 5/5, hard tasks 0/5 and there's a few flappers that settle the score. Five runs also smooths out task-run variance. 

> Show some extreme examples of the same task 

Web Search Tool -> Reward Hacking.


## Trace Integrity



## Benchmark Trace Analysis

This is harder than it looks. 

## Researchers.

Xiaomi Team --> Lets return quickly to 

Livestreamed a training run.

https://mimo.xiaomi.com/rl/

It turns out that post-training Reinforcement Learning environments look a lot like benchmark environments. Environments are created (at vast scale), tasks run and verified. However rather than just report a score, the trajectory is used to train the model. So the strongest behaviours survive - that's how models.


((letta-ai))

## Harness Integrity Issues

- Enabled web search
- Session files/traces not recording the results of web searches
- Traces not recording pre-compaction material.
- 

## Benchmark Lures

- 

## atif-scan 



## Not the same model. 

A reported number is ideally run under repeatable controlled conditions. Same timeouts, same time-of-day. 
You are fighting the public APIs .

Include the mutli-provider deepseek problem. 

## Anyone can make a claim

> Claims without evidence are marketing - 
> Evidence without reproducability is a trick

(Diagram of env create, agents setup, task, verifier)


## Reinforcement Learning

So it turns out that benchmarking, and model training have a lot of overlaps. Xiamoi team training dashboard.

Lets take a short detour and talk a little about how modern models are trained. .

## Time Limit Skewing

Find the example of the Harness with the laptop close problem.

## Harness Misreporting

Weird incentives. 


## Reward Hacking - Stage 1

mini-swe-agent and terminus 2 

What do we mean when we talk about a reward hack? The model, given a task can reason and use tools to solve it. But the real aim is to satisfy the verifier.

Install tesseract-ocr

Deepseek using an ASCII renderer to play chess.

atif-scan

## Distillation

## 


Who are they for?
Behind the number.
Trace Integrity.
Harder than it looks
Reward Hacking
Reinforcement Learning
Escaping
