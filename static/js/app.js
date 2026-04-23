// SupportMailAgent Frontend - Email Dashboard
// Handles email composition, submission, and workflow visualization

class EmailDashboard {
    constructor() {
        this.emails = [];
        this.selectedEmailId = null;
        this.templates = {};

        this.initElements();
        this.loadTemplates();
        this.attachEventListeners();
        this.loadEmails();

        // Auto-refresh emails every 2 seconds
        setInterval(() => this.loadEmails(), 2000);
    }

    initElements() {
        this.form = document.getElementById('emailForm');
        this.senderInput = document.getElementById('sender');
        this.subjectInput = document.getElementById('subject');
        this.bodyInput = document.getElementById('body');
        this.submitStatus = document.getElementById('submitStatus');
        this.emailList = document.getElementById('emailList');
        this.emailCount = document.getElementById('emailCount');
        this.detailsContent = document.getElementById('detailsContent');
    }

    loadTemplates() {
        const templatesJson = document.getElementById('emailTemplates');
        if (templatesJson) {
            this.templates = JSON.parse(templatesJson.textContent);
        }
    }

    attachEventListeners() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));

        // Template buttons
        document.querySelectorAll('.btn-template').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const templateKey = btn.getAttribute('data-template');
                this.loadTemplate(templateKey);
            });
        });
    }

    loadTemplate(key) {
        if (this.templates[key]) {
            const template = this.templates[key];
            this.senderInput.value = template.sender;
            this.subjectInput.value = template.subject;
            this.bodyInput.value = template.body;
            this.bodyInput.focus();
        }
    }

    async handleSubmit(e) {
        e.preventDefault();

        const email = {
            sender: this.senderInput.value,
            subject: this.subjectInput.value,
            body: this.bodyInput.value,
        };

        this.showStatus('Sending email...', 'loading');
        this.form.querySelector('button').disabled = true;

        try {
            const response = await fetch('/emails/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(email),
            });

            if (!response.ok) {
                const error = await response.json();
                const detail = error.detail;
                const message = Array.isArray(detail)
                    ? detail.map(e => e.msg).join(', ')
                    : (detail || 'Failed to process email');
                throw new Error(message);
            }

            const result = await response.json();
            this.showStatus('✓ Email sent successfully!', 'success');

            // Clear form
            this.form.reset();

            // Reload emails
            setTimeout(() => this.loadEmails(), 500);
        } catch (error) {
            console.error('Error:', error);
            this.showStatus(`✗ Error: ${error.message}`, 'error');
        } finally {
            this.form.querySelector('button').disabled = false;
        }
    }

    showStatus(message, type) {
        this.submitStatus.textContent = message;
        this.submitStatus.className = `status-message ${type}`;

        if (type === 'success' || type === 'error') {
            setTimeout(() => {
                this.submitStatus.className = 'status-message';
                this.submitStatus.textContent = '';
            }, 3000);
        }
    }

    async loadEmails() {
        try {
            const response = await fetch('/emails/');
            if (!response.ok) throw new Error('Failed to load emails');

            const data = await response.json();
            const emailsArray = Object.entries(data.emails || {}).map(([id, content]) => ({
                id,
                ...content,
            }));

            this.emails = emailsArray.sort((a, b) =>
                new Date(b.timestamp) - new Date(a.timestamp)
            );

            this.updateEmailList();
            this.updateEmailCount();

            // Keep selected email visible
            if (this.selectedEmailId) {
                const selected = this.emails.find(e => e.id === this.selectedEmailId);
                if (!selected) {
                    this.selectedEmailId = null;
                    this.showEmptyDetails();
                }
            }
        } catch (error) {
            console.error('Error loading emails:', error);
        }
    }

    updateEmailCount() {
        const count = this.emails.length;
        this.emailCount.textContent = `${count} email${count !== 1 ? 's' : ''}`;
    }

    updateEmailList() {
        if (this.emails.length === 0) {
            this.emailList.innerHTML = `
                <div class="empty-state">
                    <p>No emails yet. Compose one to get started!</p>
                </div>
            `;
            return;
        }

        this.emailList.innerHTML = this.emails.map(email => `
            <div class="email-item ${this.selectedEmailId === email.id ? 'active' : ''} ${email.workflow.should_escalate ? 'escalated' : ''}"
                 data-email-id="${email.id}">
                <div class="email-sender">${this.escapeHtml(email.input.sender)}</div>
                <div class="email-subject">${this.escapeHtml(email.input.subject)}</div>
                <div class="email-preview">${this.escapeHtml(email.input.body.substring(0, 60))}</div>
                <div>
                    ${email.workflow.should_escalate ? '<span class="email-badge badge-escalated">🚨 Escalated</span>' : '<span class="email-badge badge-auto">✓ Auto-Reply</span>'}
                </div>
            </div>
        `).join('');

        // Add click listeners
        this.emailList.querySelectorAll('.email-item').forEach(item => {
            item.addEventListener('click', () => {
                const emailId = item.getAttribute('data-email-id');
                this.selectEmail(emailId);
            });
        });
    }

    selectEmail(emailId) {
        this.selectedEmailId = emailId;
        this.updateEmailList();
        this.showEmailDetails(emailId);
    }

    async showEmailDetails(emailId) {
        const email = this.emails.find(e => e.id === emailId);
        if (!email) {
            this.showEmptyDetails();
            return;
        }

        const workflow = email.workflow;
        const input = email.input;
        const response = email.email;

        // Format confidence percentage
        const confidencePercent = Math.round(workflow.confidence * 100);
        const confidenceLevel =
            confidencePercent >= 70 ? 'high' :
            confidencePercent >= 40 ? 'medium' :
            'low';

        // Format KB results
        const kbResultsHtml = workflow.kb_results && workflow.kb_results.length > 0
            ? `<div class="kb-results">
                ${workflow.kb_results.slice(0, 3).map(result => `
                    <div class="kb-result">
                        <span class="kb-result-id">${this.escapeHtml(result.id)}</span>
                        <span class="kb-result-score">${Math.round(result.similarity * 100)}% match</span>
                        <div class="kb-result-preview">${this.escapeHtml(result.content.substring(0, 100))}...</div>
                    </div>
                `).join('')}
              </div>`
            : '<div class="kb-results"><em style="color: var(--gray-500);">No KB results found</em></div>';

        this.detailsContent.innerHTML = `
            <!-- Input Email -->
            <div class="details-section">
                <div class="section-title">📧 Customer Email</div>

                <div class="detail-row">
                    <span class="detail-label">From:</span>
                    <span class="detail-value">${this.escapeHtml(input.sender)}</span>
                </div>

                <div class="detail-row">
                    <span class="detail-label">Subject:</span>
                    <span class="detail-value">${this.escapeHtml(input.subject)}</span>
                </div>

                <div class="detail-row">
                    <span class="detail-label">Message:</span>
                </div>
                <div class="response-box">${this.escapeHtml(input.body)}</div>
            </div>

            <!-- Workflow Execution -->
            <div class="details-section">
                <div class="section-title">⚙️ Workflow Execution</div>

                <div class="detail-row">
                    <span class="detail-label">Intent:</span>
                    <span class="detail-value">${this.escapeHtml(workflow.intent || 'unknown')}</span>
                </div>

                <div class="detail-row">
                    <span class="detail-label">Confidence:</span>
                    <span class="detail-value">${confidencePercent}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceLevel}" style="width: ${confidencePercent}%"></div>
                    </div>
                </div>

                <div class="detail-row">
                    <span class="detail-label">Status:</span>
                    <span class="detail-value ${workflow.should_escalate ? 'escalated' : 'auto'}">
                        ${workflow.should_escalate ? '🚨 Escalated to Human' : '✓ Auto-Reply Sent'}
                    </span>
                </div>

                ${workflow.followup_scheduled ? `
                    <div class="detail-row">
                        <span class="detail-label">Follow-up:</span>
                        <span class="detail-value">📅 Scheduled (24h)</span>
                    </div>
                ` : ''}
            </div>

            <!-- Knowledge Base Results -->
            <div class="details-section">
                <div class="section-title">🔍 KB Search Results</div>
                ${kbResultsHtml}
            </div>

            <!-- AI Response -->
            <div class="details-section">
                <div class="section-title">💬 AI Response</div>
                <div class="detail-row">
                    <span class="detail-label">To:</span>
                    <span class="detail-value">${this.escapeHtml(response.recipient)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Subject:</span>
                    <span class="detail-value">${this.escapeHtml(response.subject)}</span>
                </div>
                <div class="response-box">${this.escapeHtml(response.body)}</div>
            </div>

            <!-- Metadata -->
            <div class="details-section">
                <div class="section-title">📋 Metadata</div>
                <div class="detail-row">
                    <span class="detail-label">Email ID:</span>
                    <span class="detail-value" style="font-family: monospace; font-size: 10px;">${this.escapeHtml(email.id)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Time:</span>
                    <span class="detail-value">${new Date(email.timestamp).toLocaleString()}</span>
                </div>
            </div>
        `;
    }

    showEmptyDetails() {
        this.detailsContent.innerHTML = `
            <div class="empty-state">
                <p>Select an email to view details</p>
            </div>
        `;
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new EmailDashboard();
});
