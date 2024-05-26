import json

import torch
from torch.utils.data import DataLoader, Dataset

import accelerate 

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm

from rouge_score import rouge_scorer

def encode_and_tokenize_instance(instance, tokenizer):
	fact = instance['fact1']
	stem = instance['question']['stem']
	choices = ' '.join([f"[{choice['label']}] {choice['text']}" for choice in instance['question']['choices']])
	answer = instance['answerKey']
	prompt = f"[START] {fact} {stem} {choices} [ANSWER] {answer}"

	encoded = tokenizer(prompt, padding = "max_length", truncation = True, max_length = 512, return_tensors='pt')
	labels = encoded['input_ids'].clone()
	labels[labels == tokenizer.pad_token_id] = -100
	
	return {'input_ids':encoded['input_ids'].squeeze(), 
		    'attention_mask':encoded['attention_mask'].squeeze(),
		    'labels':labels.squeeze()}



class CustomDataset(Dataset):
	def __init__(self, data, tokenizer):
		self.data = data
		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.data)

	def __getitem__(self,idx):
		instance = self.data[idx]
		encoded_instance = encode_and_tokenize_instance(instance, self.tokenizer)
		return encoded_instance


def load_and_tokenize_data(filepath, tokenizer):
	with open(filepath, 'r') as f:
		data = [json.loads(line) for line in f]
	return CustomDataset(data, tokenizer)

def train(train_dataloader, num_epochs, model, device, optimizer, scheduler,loss_fn):
	for epoch in range(num_epochs):
		model.train()
		for batch in tqdm(train_dataloader):
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			optimizer.zero_grad()
			outputs = model(input_ids,attention_mask = attention_mask,labels= labels)

			shift_logits = outputs.logits[...,:,:].contiguous()
			shift_labels = labels[...,:].contigous()
			batch_indices = torch.arange(shift_logits.size(0)).to(device)
			last_index = (labels != -100).sum(dim=1).to(device)

			last_token_logits = shift_logits[batch_indices, last_index - 2, :]
			last_token_labels = shift_labels[batch_indices, last_index -1]

			loss = loss_fn(last_token_logits, last_token_labels)

			loss.backward()
			optimizer.step()
			scheduler.step()
		print(f"Epoch: {epoch}, Loss: {loss.item()}")

	model.save_pretrained('./custom_gpt2_model')
	tokenizer.save_pretrained('./custom_gpt2_model')


def compute_rouge(pred, true):
	scorer = rouge_score.RougeScorer(['rougeL'], use_stemmer = True)
	scores = scorer.score(pred, true)
	return scores['rougeL'].fmeasure

def evaluate(test_path, test_dataloader, model, device):
	with open(test_path, 'r') as f:
		test_data = [json.loads(line) for line in f]

	model.eval()
	predictions = []
	with torch.no_grad():
		for batch in test_dataloader:
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)

			outputs = model.generate(input_ids = input_ids, attention_mask = attention_mask, max_new_tokens = 50)
			predictions.extend(outputs)

	correct_answers = 0
	total_answers = 0

	for idx, instance in enumerate(test_data):
		input_ids = predictions[idx]
		decoded_output = tokenizer.decode(input_ids, skip_special_tokens = True)

		choices = instance['question']['choices']
		answer = instance['answerKey']

		choice_texts = [choice['text'] for choice in choices]
		choice_labels = [choice['label'] for choice in choices]

		rouge_scores = [compute_rouge(decoded_output, choice_text) for choice_text in choice_texts]
		predicted_label = choice_labels[rouge_scores.index(max(rouge_scores))]

		if predicted_label == answer:
			correct_answers +=1
		total_answers += 1

	accuracy = correct_answers/total_answers
	print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
	train_path = "train_complete.jsonl"
	test_path = "test_complete.jsonl"

	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	tokenizer.pad_token = tokenizer.eos_token

	train_data = load_and_tokenize_data(train_path, tokenizer)
	test_data = load_and_tokenize_data(test_path, tokenizer)

	train_dataloader = DataLoader(train_data, batch_size = 4, shuffle = True)
	test_dataloader = DataLoader(test_data, batch_size = 4, shuffle = False)

	device = torch.device("cuda" if cuda.is_available() else "cpu")

	model = GPT2LMHeadModel.from_pretrained("gpt2")
	model.resize_token_embeddings(len(tokenizer))
	model.to(device)

	num_epochs = 3
	optmizer = AdamW(model.parameters(), lr = 5e-5)
	total_steps = len(train_dataloader) * num_epochs
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

	train(train_dataloader, num_epochs, model, device, optimizer, scheduler, torch.nn.CrossEntropyLoss())

	evaluate(test_path, test_dataloader, model, device)

