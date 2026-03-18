# Deploy to Render.com (FREE)

Your FastAPI backend is ready to deploy. Follow these 5 steps.

## Step 1: Push to GitHub

```bash
git add .
git commit -m "Add Render deployment config"
git push origin main
```

## Step 2: Create Render Service

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **"Email support"** project (your existing project)
3. Click **"+ New"** → **"Web Service"**

## Step 3: Connect GitHub Repo

1. Select **"Connect a repository"**
2. Search for **"SupportMailAgent"** (your repo)
3. Click **"Connect"**

## Step 4: Configure Service

Fill in these settings:

| Setting | Value |
|---------|-------|
| **Name** | `support-mail-api` |
| **Runtime** | `Python 3` |
| **Build Command** | `bash build.sh` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Free Tier** | ✅ Yes |

## Step 5: Add Environment Variables

Click **"Add Environment Variable"** and add:

```
ANTHROPIC_API_KEY = sk-ant-your-api-key-here
```

Then click **"Create Web Service"** and wait ~2 minutes for deployment.

## Step 6: Get Your Live URL

Once deployed, you'll see:
```
Your service is live at: https://support-mail-api-xxxxx.onrender.com
```

**Copy this URL!**

## Step 7: Update Streamlit

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click your "Support Mail Agent" app
3. Click **⋮ Settings** (top right)
4. Go to **Secrets**
5. Update to:
   ```toml
   API_BASE_URL = "https://support-mail-api-xxxxx.onrender.com"
   ```
6. Save and Streamlit will auto-redeploy

## Step 8: Test It!

1. Go to your Streamlit app
2. Enter a test email
3. Click "🚀 Process Email"
4. It should work! ✅

## Important Notes

**Free tier limits:**
- Your API will sleep after 15 minutes of inactivity
- First request might take 10-20 seconds to wake up
- This is fine for demos/resumes

**To prevent sleeping (optional):**
- Upgrade to paid tier ($7/month) - but NOT necessary
- Or use a ping service to keep it awake

## Troubleshooting

**API shows "service down"?**
- Wait 2-3 minutes, it might still be deploying
- Check Render dashboard for build errors

**"Cannot connect" from Streamlit?**
- Make sure you copied the URL correctly
- Try the URL in browser: `https://your-url/health`

**ANTHROPIC_API_KEY error?**
- Verify your API key is correct in Render secrets
- Make sure it starts with `sk-ant-`

## You're Done! 🎉

Now you have:
- ✅ FastAPI backend running on Render (FREE)
- ✅ Streamlit frontend on Streamlit Cloud (FREE)
- ✅ Live demo for your resume/portfolio
- ✅ No paid charges!
